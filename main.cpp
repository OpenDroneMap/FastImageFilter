#include <iostream>
#include <limits>
#include <cmath>
#include <omp.h>

#include "vendor/cxxopts.hpp"
#include "vendor/mf2d/filter.hpp"

#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()

#include "utils.hpp"

int main(int argc, char **argv) {
    cxxopts::Options options("fastimagefilter", "Fast image filtering of georeferenced raster DEMs");
    options.add_options()
        ("i,input", "Input raster DEM (.tif)", cxxopts::value<std::string>())
        ("o,output", "Output raster DEM (.tif)", cxxopts::value<std::string>()->default_value("output.tif"))
        ("r,radius", "Radius of median filter (pixels)", cxxopts::value<int>()->default_value("9"))
        ("w,window-size", "Window size of raster", cxxopts::value<int>()->default_value("512"))
        ("c,co", "GDAL creation options", cxxopts::value<std::vector<std::string> >()->default_value(""))
        ("h,help", "Print usage")
        ;
    options.parse_positional({ "input" });
    options.positional_help("[input]");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help") || !result.count("input")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    try {
        const auto inputFilename = result["input"].as<std::string>();
        const auto outputFilename = result["output"].as<std::string>();
        const auto windowSize = result["window-size"].as<int>();
        const auto radius = result["radius"].as<int>();
        const auto cOpts = result["co"].as<std::vector<std::string> >();
        
        char **papszOptions = NULL;
        for (auto &co : cOpts){
            std::vector<std::string> kv = split(co, "=");
            if (kv.size() == 2){
                papszOptions = CSLSetNameValue(papszOptions, kv[0].c_str(), kv[1].c_str());
            }else{
                std::cerr << "Invalid --co " << co << std::endl;
                exit(1);
            }
        }

        int maxConcurrency = omp_get_max_threads();

        GDALDataset  *dataset;
        GDALAllRegister();
        dataset = (GDALDataset *) GDALOpen( inputFilename.c_str(), GA_ReadOnly );
        if( dataset == NULL ) throw std::runtime_error("Cannot open " + inputFilename);

        int width = dataset->GetRasterXSize();
        int height = dataset->GetRasterYSize();
        int hasNoData = FALSE;

        GDALRasterBand *band = dataset->GetRasterBand(1);
        double nodata = band->GetNoDataValue(&hasNoData);

        std::cout << "Raster size is " << width << "x" << height << std::endl;

        GDALDataset *dst = dataset->GetDriver()->Create(outputFilename.c_str(), width, height, 1, GDT_Float32, papszOptions);
        if (dataset->GetSpatialRef() != nullptr){
            dst->SetSpatialRef(dataset->GetSpatialRef());
            double geotransform[6];
            dataset->GetGeoTransform(geotransform);
            dst->SetGeoTransform(geotransform);
        }

        GDALRasterBand *writeBand = dst->GetRasterBand(1);
        if (hasNoData){
            writeBand->SetNoDataValue(nodata);
        }

        int blockSizeX = (std::min)(windowSize, width);
        int blockSizeY = (std::min)(windowSize, height);

        int subX = static_cast<int>(std::ceil(static_cast<double>(width) / static_cast<double>(blockSizeX)));
        int subY = static_cast<int>(std::ceil(static_cast<double>(height) / static_cast<double>(blockSizeY)));
        int numBlocks = subX * subY;

        omp_lock_t readLock;
        omp_init_lock(&readLock);

        omp_lock_t writeLock;
        omp_init_lock(&writeLock);

        int rasterDataBlocks = std::min(maxConcurrency, numBlocks);
        float *rasterData = new float[rasterDataBlocks * (blockSizeX + 1) * (blockSizeY + 1)];
        uint8_t *nodataBuffer = nullptr;
        if (hasNoData) nodataBuffer = new uint8_t[rasterDataBlocks * (blockSizeX + 1) * (blockSizeY + 1)];
        
        float nanValue = std::numeric_limits<double>::quiet_NaN();
        size_t pxCount = blockSizeX * blockSizeY;

        std::cout << "Smoothing...";

        #pragma omp parallel for collapse(2)
        for (int blockX = 0; blockX < subX; blockX++){
            for (int blockY = 0; blockY < subY; blockY++){
                int sizeX = blockX == subX - 1 ? width - (blockSizeX * blockX) : blockSizeX;
                int sizeY = blockY == subY - 1 ? height - (blockSizeY * blockY) : blockSizeY;
            
                int t = omp_get_thread_num();
                int xOffset = blockX * blockSizeX;
                int yOffset = blockY * blockSizeY;

                float *rasterPtr = rasterData + t * (blockSizeX + 1) * (blockSizeY + 1);
                uint8_t *nodataPtr;
                if (hasNoData) {
                    nodataPtr = nodataBuffer + t * (blockSizeX + 1) * (blockSizeY + 1);
                    memset(nodataPtr, 0, pxCount);
                }

                omp_set_lock(&readLock);
                if (band->RasterIO( GF_Read, xOffset, yOffset, sizeX, sizeY,
                                   rasterPtr, blockSizeX, blockSizeY, GDT_Float32, 0, 0 ) == CE_Failure){
                    std::cerr << "Cannot access raster data" << std::endl;
                    exit(EXIT_FAILURE);
                }
                omp_unset_lock(&readLock);

                if (hasNoData){
                    for (size_t i = 0; i < pxCount; i++){
                        if (rasterPtr[i] == nodata){
                            rasterPtr[i] = nanValue;
                            nodataPtr[i] = 1;
                        }
                    }
                }

                median_filter_2d<float>(blockSizeX, blockSizeY, radius, radius, 0, rasterPtr, rasterPtr);

                if (hasNoData){
                    for (size_t i = 0; i < pxCount; i++){
                        if (nodataPtr[i]){
                            rasterPtr[i] = nodata;
                        }
                    }
                }

                omp_set_lock(&writeLock);
                if (writeBand->RasterIO( GF_Write, xOffset, yOffset, sizeX, sizeY,
                                   rasterPtr, blockSizeX, blockSizeY, GDT_Float32, 0, 0 ) == CE_Failure){
                    std::cerr << "Cannot write raster data" << std::endl;
                    exit(EXIT_FAILURE);
                }
                omp_unset_lock(&writeLock);

                std::cout << ".";
                std::flush(std::cout);
            }
        }

        delete dst;
        delete dataset;

        std::cout << " done" << std::endl << "Wrote " << outputFilename << std::endl;
    }catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

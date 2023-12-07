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
        if (!dataset->GetSpatialRef()->IsEmpty()){
            dst->SetSpatialRef(dataset->GetSpatialRef());
            double geotransform[6];
            dataset->GetGeoTransform(geotransform);
            dst->SetGeoTransform(geotransform);
        }

        GDALRasterBand *writeBand = dst->GetRasterBand(1);
        if (hasNoData){
            writeBand->SetNoDataValue(nodata);
        }

        float nanValue = std::numeric_limits<double>::quiet_NaN();
        size_t pxCount = width * height;

        float *rasterData = new float[(width + 1) * (height + 1)];
        uint8_t *nodataBuffer = nullptr;
        if (hasNoData) {
            nodataBuffer = new uint8_t[(width + 1) * (height + 1)];
            memset(nodataBuffer, 0, pxCount);
        }
        
        std::cout << "Smoothing... ";

        if (band->RasterIO( GF_Read, 0, 0, width, height,
                            rasterData, width, height, GDT_Float32, 0, 0 ) == CE_Failure){
            std::cerr << "Cannot access raster data" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (hasNoData){
            for (size_t i = 0; i < pxCount; i++){
                if (rasterData[i] == nodata){
                    rasterData[i] = nanValue;
                    nodataBuffer[i] = 1;
                }
            }
        }

        median_filter_2d<float>(width, height, radius, radius, 0, rasterData, rasterData);

        if (hasNoData){
            for (size_t i = 0; i < pxCount; i++){
                if (nodataBuffer[i]){
                    rasterData[i] = nodata;
                }
            }
        }

        if (writeBand->RasterIO( GF_Write, 0, 0, width, height,
                            rasterData, width, height, GDT_Float32, 0, 0 ) == CE_Failure){
            std::cerr << "Cannot write raster data" << std::endl;
            exit(EXIT_FAILURE);
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

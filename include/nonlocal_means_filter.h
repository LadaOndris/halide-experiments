
#ifndef HALIDE_EXPERIMENTS_NONLOCAL_MEANS_FILTER_H
#define HALIDE_EXPERIMENTS_NONLOCAL_MEANS_FILTER_H

#include <cstdint>
#include "Halide.h"
#include "HalidePipeline.h"

using namespace Halide;

class NonlocalMeansFilter : public HalidePipeline {
private:
    Buffer<uint8_t> input;
    int patchSize;
    int searchWindowSize;

    float h = 0.1;
    float weighingGaussianSigma = 1.5f;

    void implement();

    static Func createGaussian(int width, int height, float sigma);

public:
    // Coordinates of point 1
    Var x, y;
    // Coordinates of point 2
    Var a, b;
    // Offsets within the patch
    Var i, j;

    Func clamped;
    Func gaussian;
    Func weightedPixelDist;
    Func neighborhoodDifference;
    Func areDifferentPoints;
    Func neighborhoodWeight;
    Func weightsSum;
    Func newPixelValues;
    Func newPixelValuesNormalized;

    explicit NonlocalMeansFilter(const Buffer<uint8_t>& input, int patchSize, int searchWindowSize);

    bool scheduleForGPU() override;

    void scheduleForCPU() override;

};

#endif //HALIDE_EXPERIMENTS_NONLOCAL_MEANS_FILTER_H

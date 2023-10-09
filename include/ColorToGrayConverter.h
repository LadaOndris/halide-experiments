
#ifndef HALIDE_EXPERIMENTS_COLORTOGRAYCONVERTER_H
#define HALIDE_EXPERIMENTS_COLORTOGRAYCONVERTER_H


#include <cstdint>
#include "Halide.h"
#include "HalidePipeline.h"

using namespace Halide;

class ColorToGrayConverter : public HalidePipeline {
private:
    Buffer<uint8_t> input;

    void implement();

public:
    Var x, y, c;
    Func convertToGray;

    explicit ColorToGrayConverter(const Buffer<uint8_t> &input);

    bool scheduleForGPU() override;

    void scheduleForCPU() override;
};

#endif //HALIDE_EXPERIMENTS_COLORTOGRAYCONVERTER_H

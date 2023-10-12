
#ifndef HALIDE_EXPERIMENTS_HALIDEPIPELINE_H
#define HALIDE_EXPERIMENTS_HALIDEPIPELINE_H

#include "Halide.h"

using namespace Halide;

class HalidePipeline {
public:
    Func result;

    virtual bool scheduleForGPU() = 0;

    virtual void scheduleForCPU() = 0;
};


#endif //HALIDE_EXPERIMENTS_HALIDEPIPELINE_H

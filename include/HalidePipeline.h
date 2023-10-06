
#ifndef HALIDE_EXPERIMENTS_HALIDEPIPELINE_H
#define HALIDE_EXPERIMENTS_HALIDEPIPELINE_H

class HalidePipeline {
public:
    virtual bool scheduleForGPU() = 0;

    virtual void scheduleForCPU() = 0;
};


#endif //HALIDE_EXPERIMENTS_HALIDEPIPELINE_H

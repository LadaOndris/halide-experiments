
#include "ColorToGrayConverter.h"
#include "gpu.h"

ColorToGrayConverter::ColorToGrayConverter(const Buffer<uint8_t> &input)
        : input(input),
          x("x"), y("y"), c("c"),
          convertToGray("convertToGray") {
    implement();
}

void ColorToGrayConverter::implement() {
    result(x, y) = cast<uint8_t>(0.299f * input(x, y, 0) +
                                 0.587f * input(x, y, 1) +
                                 0.114f * input(x, y, 2));
}

void ColorToGrayConverter::scheduleForCPU() {
    Var x_inner, y_inner, x_outer, y_outer, tile_index;
    result.vectorize(x, 4)
            .parallel(y);
}

bool ColorToGrayConverter::scheduleForGPU() {
    Target target = find_gpu_target();
    if (!target.has_gpu_feature()) {
        return false;
    }
    Var xi, yi, xo, yo;
    result.gpu_threads(x, y);

    printf("Target: %s\n", target.to_string().c_str());
    result.compile_jit(target);
    return true;
}



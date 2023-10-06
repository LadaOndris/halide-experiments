// Halide tutorial lesson 2: Processing images

// This lesson demonstrates how to pass in input images and manipulate
// them.

// On linux, you can compile and run it like so:
// g++ lesson_02*.cpp -g -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o lesson_02 -std=c++17
// LD_LIBRARY_PATH=<path/to/libHalide.so> ./lesson_02

// On os x:
// g++ lesson_02*.cpp -g -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> -L <path/to/libHalide.so> -lHalide `libpng-config --cflags --ldflags` -ljpeg -o lesson_02 -std=c++17
// DYLD_LIBRARY_PATH=<path/to/libHalide.dylib> ./lesson_02

// If you have the entire Halide source tree, you can also build it by
// running:
//    make tutorial_lesson_02_input_image
// in a shell with the current directory at the top of the halide
// source tree.

// The only Halide header file you need is Halide.h. It includes all of Halide.
#include "Halide.h"

// Include some support code for loading pngs.
#include "include/stb_image.h"
#include "include/stb_image_write.h"
//using namespace Halide::Tools;
using namespace Halide;


int processHalide() {
    int width, height, nrChannels;
    unsigned char *data = stbi_load("images/lena_grayscale.jpg", &width, &height, &nrChannels, 0);
    if (!data) {
        // Handle the error (e.g., print an error message)
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return 1; // Exit with an error code
    }

    Halide::Buffer<uint8_t> input(data, width, height);

    Func clamped("clamped");
    clamped = BoundaryConditions::repeat_edge(input);

    Var x("x"), y("y"), a("a"), b("b");

    int outer_neighborhood_size = 11;
    int inner_neighborhood_size = 5;

    RDom r_inner(-inner_neighborhood_size / 2, inner_neighborhood_size,
                 -inner_neighborhood_size / 2, inner_neighborhood_size);
    RDom r_outer(-outer_neighborhood_size / 2, outer_neighborhood_size,
                 -outer_neighborhood_size / 2, outer_neighborhood_size);

    Func neighborhoodSum("neighborhoodSum");
    neighborhoodSum(x, y) = sum(clamped(x + r_inner.x, y + r_inner.y));

    // The difference between individual pixels
    Func pixelDifference("pixelDifference");
    pixelDifference(x, y, a, b) = cast<uint8_t>(absd(clamped(x, y), clamped(a, b)));

    // The difference between two patches
    Func neighborhoodDifference("neighborhoodDifference");
    neighborhoodDifference(x, y, a, b) = sum(pixelDifference(x + r_inner.x, y + r_inner.y,
                                                             a + r_inner.x, b + r_inner.y));

    // The sum of the sums of differences of the current patch and secondary patches
    Func differencesSum("differencesSum");
    differencesSum(x, y) += neighborhoodDifference(x, y, x + r_outer.x, y + r_outer.y);

    Func result("result");
    result(x, y) = cast<uint8_t>(differencesSum(x, y));

    result.compute_root();

    Halide::Buffer<uint8_t> output =
            result.realize({input.width(), input.height()});

    printf("Pseudo-code for the schedule:\n");
    result.print_loop_nest();
    printf("\n");

    auto *outputData = new uint8_t[width * height * nrChannels];
    memcpy(outputData, output.data(), width * height * nrChannels);
    stbi_write_png("outputs/output.png", width, height, nrChannels, outputData, width * nrChannels);

    delete[] outputData;
    stbi_image_free(data);

    printf("Success!\n");
}

int main(int argc, char **argv) {
    try {
        return processHalide();
    } catch (CompileError &e) {
        std::cout << e.what() << std::endl;
    }  catch (RuntimeError &e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
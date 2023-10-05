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
    unsigned int search_window_size = 10;
    unsigned int neighborhood_size = 8;


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

    // Sums the value in the neighborhood
    Expr neighborhood_expr = static_cast<Expr>(neighborhood_size);
    Expr half_neighborhood_expr = static_cast<Expr>(neighborhood_size / 2);
    Expr squared_neighborhood = static_cast<Expr>(neighborhood_size * neighborhood_size);

    Func nonLocalMeansFunc("nonLocalMeansFunc");
    nonLocalMeansFunc(x, y) = cast<uint32_t>(0);
    RDom r(-half_neighborhood_expr, neighborhood_expr, -half_neighborhood_expr, neighborhood_expr);
    nonLocalMeansFunc(x, y) += cast<uint32_t>(clamped(x + r.x, y + r.y));
    nonLocalMeansFunc(x, y) /= squared_neighborhood;


    Func result("result");
    result(x, y) = cast<uint8_t>(nonLocalMeansFunc(x, y));

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
    }

    return 0;
}
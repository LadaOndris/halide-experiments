#include "Halide.h"

// Include some support code for loading pngs.
#include "include/stb_image.h"
#include "include/stb_image_write.h"
//using namespace Halide::Tools;
using namespace Halide;


int processHalide() {
    // Define the input image dimensions

    int width, height, nrChannels;
    unsigned char *data = stbi_load("images/lena_grayscale.jpg", &width, &height, &nrChannels, 0);
    if (!data) {
        // Handle the error (e.g., print an error message)
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return 1; // Exit with an error code
    }

    Halide::Buffer<uint8_t> input(data, width, height);

    // Define a Func to compute the difference between pixel pairs
    Var x, y, a, b;
    Func difference("difference");
    // RDom r(0, width, 0, height); // Iterate over all pixel pairs

    difference(x, y, a, b) = cast<uint8_t>(absd(input(x, y), input(a, b)));

    difference.compute_root();

    auto output = difference.realize({input.width(), input.height(), input.width(), input.height()});
}

int main() {
    try {
        return processHalide();
    } catch (CompileError &e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}

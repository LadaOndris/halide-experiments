#include <random>
#include <utility>
#include "Halide.h"
#include "../lib/stb/stb_image.h"
#include "../lib/stb/stb_image_write.h"

using namespace Halide;

Expr gaussian_random(Expr sigma) {
    return (random_float() + random_float() + random_float() - 1.5f) * 2 * std::move(sigma);
}

Func getNoisyImageFunc(float sigma) {
    Var x, y;

    Func noise;
    noise(x, y) = abs(gaussian_random(sigma));

    // Create a Halide Expr for the noise
    Func input("input");
    input(x, y) = cast<uint8_t>(max(min(10 * x + noise(x, y), 255), 0));

    return input;
}

Buffer<uint8_t> createNoisyImage(int size, float gaussianNoiseSigma) {
    Func noisyInput = getNoisyImageFunc(gaussianNoiseSigma);
    Buffer<uint8_t> buffer = noisyInput.realize({size, size});
    return buffer;
}

Buffer<uint8_t> loadImageFromFile(const std::string &filePath) {
    int width;
    int height;
    int channels;
    unsigned char *data = stbi_load(filePath.c_str(), &width, &height, &channels, 0);
    if (!data) {
        // Handle the error (e.g., print an error message)
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        // TODO: raise exception
    }
    // TODO: free memory
    // stbi_image_free(data);
    Buffer<uint8_t> buffer;
    if (channels > 1) {
        buffer = Buffer<uint8_t>::make_interleaved(data, width, height, channels);
    } else {
        buffer = Buffer<uint8_t>(data, width, height);
    }
    // Signal for the GPU that the buffer's changed.
    buffer.set_host_dirty();
    return buffer;
}

void saveImageToFile(Buffer<uint8_t> image, const std::string &targetFilePath) {
//    int numElements = image.width() * image.height() * image.channels();
//    auto *inputData = new uint8_t[numElements];
//    memcpy(inputData, image.data(), numElements);
    int result = stbi_write_png(targetFilePath.c_str(), image.width(), image.height(), image.channels(),
                                image.data(), image.width() * image.channels());
    if (result == 0) {
        std::cerr << "Error: Failed to save image to file." << std::endl;
    }
//    delete[] inputData;
}

#include "Halide.h"
#include <random>
#include <chrono>

#include "include/stb_image.h"
#include "include/stb_image_write.h"
#include "nonlocal_means_filter.h"

using namespace Halide;

Expr gaussian_random(Expr sigma) {
    return (random_float() + random_float() + random_float() - 1.5f) * 2 * sigma;
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

Buffer<uint8_t> loadImageFromFile(std::string filePath) {
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
    Halide::Buffer<uint8_t> buffer(data, width, height);
    return buffer;
}

Buffer<uint8_t> createNoisyImage(int size, float gaussianNoiseSigma) {
    Func noisyInput = getNoisyImageFunc(gaussianNoiseSigma);
    Buffer<uint8_t> buffer = noisyInput.realize({size, size});
    return buffer;
}

void saveImageToFile(Buffer<uint8_t> image, const std::string &targetFilePath) {
    int numElements = image.width() * image.height() * image.channels();
    auto *inputData = new uint8_t[numElements];
    memcpy(inputData, image.data(), numElements);
    stbi_write_png(targetFilePath.c_str(), image.width(), image.height(), image.channels(),
                   inputData, image.width() * image.channels());
    delete[] inputData;
}

// Custom timing function
template<typename Func>
double measureExecutionTime(Func&& func) {
    using namespace std::chrono;

    // Start the timer
    high_resolution_clock::time_point start_time = high_resolution_clock::now();

    // Execute the provided function or code block
    func();

    // Stop the timer
    high_resolution_clock::time_point end_time = high_resolution_clock::now();

    // Calculate the elapsed time in seconds
    duration<double> elapsed_seconds = duration_cast<duration<double>>(end_time - start_time);

    return elapsed_seconds.count();
}

int processHalide() {
//    int imageSize = 20;
//    float gaussianNoiseSigma = 20.f;
//    auto image = createNoisyImage(imageSize, gaussianNoiseSigma);
    auto image = loadImageFromFile("images/lena_grayscale.jpg");

    saveImageToFile(image, "outputs/input.png");

    auto realizationWidth = image.width();
    auto realizationHeight = image.height();

    int searchWindowSize = 13;
    int patchSize = 5;

    Halide::Buffer<uint8_t> outputBuffer(realizationWidth, realizationHeight);

    printf("Running pipeline on CPU:\n");
    NonlocalMeansFilter filter(image, patchSize, searchWindowSize);
    filter.scheduleForCPU();

    double executionTime = measureExecutionTime([&filter, &outputBuffer] {
        filter.result.realize(outputBuffer);
    });
    std::cout << "Execution time: " << executionTime << " seconds" << std::endl;

    printf("\n\nPseudo-code for the schedule:\n");
    filter.result.print_loop_nest();
    printf("\n");

    saveImageToFile(outputBuffer, "outputs/output.png");

    printf("Success!\n");
}

int main(int argc, char **argv) {
    try {
        return processHalide();
    } catch (CompileError &e) {
        std::cout << e.what() << std::endl;
    } catch (RuntimeError &e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}

#include "Halide.h"
#include <random>
#include <chrono>
#include <unistd.h> // for getopt

#include "include/stb_image.h"
#include "include/stb_image_write.h"
#include "nonlocal_means_filter.h"
#include "gpu.h"
#include "ColorToGrayConverter.h"

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
    if (channels > 1) {
        Halide::Buffer<uint8_t> buffer =
                Halide::Buffer<uint8_t>::make_interleaved(data, width, height, channels);
        return buffer;
    } else {
        Halide::Buffer<uint8_t> buffer(data, width, height);
        return buffer;
    }
}

Buffer<uint8_t> createNoisyImage(int size, float gaussianNoiseSigma) {
    Func noisyInput = getNoisyImageFunc(gaussianNoiseSigma);
    Buffer<uint8_t> buffer = noisyInput.realize({size, size});
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

// Custom timing function
template<typename Func>
double measureExecutionTime(Func &&func) {
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

int processHalide(const std::string &imagePath, int reps, const std::string &pipelineType) {
//    int imageSize = 20;
//    float gaussianNoiseSigma = 20.f;
//    auto image = createNoisyImage(imageSize, gaussianNoiseSigma);
    std::cout << "Finding a GPU target..." << std::endl;
    auto target = find_gpu_target();
    std::cout << "Target found: " << target.to_string().c_str() << std::endl;

    std::cout << "Preparing input image..." << std::endl;
    auto image = loadImageFromFile(imagePath);

    saveImageToFile(image, "outputs/input.png");

    auto realizationWidth = image.width();
    auto realizationHeight = image.height();

    int searchWindowSize = 13;
    int patchSize = 5;

    std::unique_ptr<HalidePipeline> pipeline;
    if (pipelineType == "colortogray") {
        assert(image.channels() == 3);
        pipeline = std::make_unique<ColorToGrayConverter>(image);
    } else if (pipelineType == "nonlocalmeans") {
        assert(image.channels() == 1);
        pipeline = std::make_unique<NonlocalMeansFilter>(image, patchSize, searchWindowSize);
    } else {
        std::cerr << "Invalid pipeline type: " << pipelineType << std::endl;
        return EXIT_FAILURE; // Return an error code
    }

    if (target.has_gpu_feature()) {
        std::cout << "Running pipeline on the GPU..." << std::endl;
        pipeline->scheduleForGPU();
    } else {
        std::cout << "Running pipeline on the CPU..." << std::endl;
        pipeline->scheduleForCPU();
    }

    auto outputBuffer = Halide::Buffer<uint8_t>(realizationWidth, realizationHeight);
    // Warm-up before measuring
    pipeline->result.realize(outputBuffer, target);

    double executionTime = measureExecutionTime([&pipeline, &outputBuffer, &reps, &target] {
        for (int i = 0; i < reps; i++) {
            pipeline->result.realize(outputBuffer, target);
        }
    });
    std::cout << "Execution time: " << (executionTime / reps) * 1000 << " ms/rep" << std::endl;

    printf("\nPseudo-code for the schedule:\n");
    pipeline->result.print_loop_nest();
    printf("\n");

    std::cout << "Saving result..." << std::endl;
    saveImageToFile(outputBuffer, "outputs/output.png");
}

struct Arguments {
    std::string imagePath;
    std::string pipelineType;
    int reps = 1;
    bool areValid = false;
};

Arguments processArguments(int argc, char **argv) {
    Arguments args;
    int opt;
    while ((opt = getopt(argc, argv, "i:r:p:")) != -1) {
        switch (opt) {
            case 'i':
                args.imagePath = optarg;
                break;
            case 'r':
                args.reps = std::stoi(optarg);
                break;
            case 'p':
                args.pipelineType = optarg;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -i <image_path> -r <reps> -p <pipeline_type>" << std::endl;
                return args;
        }
    }
    if (args.imagePath.empty() || args.pipelineType.empty()) {
        std::cerr << "Both --image and --pipeline arguments are required." << std::endl;
        return args;
    }
    args.areValid = true;
    return args;
}

int main(int argc, char **argv) {
    Arguments args = processArguments(argc, argv);
    if (!args.areValid) {
        return EXIT_FAILURE;
    }

    try {
        return processHalide(args.imagePath, args.reps, args.pipelineType);
    } catch (CompileError &e) {
        std::cout << e.what() << std::endl;
    } catch (RuntimeError &e) {
        std::cout << e.what() << std::endl;
    }

    return EXIT_SUCCESS;
}
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

using namespace std;

void listDevices() {
    ma_context context; 
    if (ma_context_init(NULL, 0, NULL, &context) != MA_SUCCESS) {
        printf("Failed to init context.");
        return;
    }
    ma_device_info* pCaptureDevices; 
    ma_uint32 captureDeviceCount; 

    if (ma_context_get_devices(&context, NULL, NULL, &pCaptureDevices, &pcaptureDeviceCount) != MA_SUCCESS) {
        printf('Failed to get devices');
        ma_context_uninit(&context); 
        return; 
    }
}

struct Audio {

    static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
    {
        ma_encoder* pEncoder = (ma_encoder*)pDevice->pUserData;
        MA_ASSERT(pEncoder != NULL);

        ma_encoder_write_pcm_frames(pEncoder, pInput, frameCount, NULL);

        (void)pOutput;
    }

    void record() {
        ma_result result;
        ma_encoder_config encoderConfig;
        ma_encoder encoder;
        ma_device_config deviceConfig;
        ma_device device;


        encoderConfig = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 2, 44100);

        if (ma_encoder_init_file("test.wav", &encoderConfig, &encoder) != MA_SUCCESS) {
            printf("Failed to initialize output file.\n");
            return;
        }

        deviceConfig = ma_device_config_init(ma_device_type_capture);
        deviceConfig.capture.format   = encoder.config.format;
        deviceConfig.capture.channels = encoder.config.channels;
        deviceConfig.sampleRate       = encoder.config.sampleRate;
        deviceConfig.dataCallback     = Audio::data_callback;
        deviceConfig.pUserData        = &encoder;

        result = ma_device_init(NULL, &deviceConfig, &device);
        if (result != MA_SUCCESS) {
            printf("Failed to initialize capture device.\n");
            return;
        }

        result = ma_device_start(&device);
        if (result != MA_SUCCESS) {
            ma_device_uninit(&device);
            printf("Failed to start device.\n");
            return;
        }

        printf("Press Enter to stop recording...\n");
        getchar();
        
        ma_device_uninit(&device);
        ma_encoder_uninit(&encoder);
    }


};

int main() {
    Audio audio; 
    audio.record(); 
    return 0;
}

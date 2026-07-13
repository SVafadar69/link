#include <iostream> 
#include "json.hpp"
#include <cstlib>

system("sudo systemctl start ollama");

using json = nlohmann::json;

void ask_ollama(std::string prompt) {
    httplib::Client cli("http://localhost:11434");


    json request_body = {
        {"model", "llama-3.2:1b"},
        {"messages": {{{"role", "user"}, {"content", prompt}}}},
        {"stream", false}

    };

    auto res = cli.Post("/v1/chat/completions", request_body.dump(), "application/json");
    if (res && res->status == 200) {
        std::cout << "Ollama response: " << res->body << std::endl;
    } else {
        std::cerr << "Failed to get response from Ollama. Status: " << (res ? res->status : -1) << std::endl;
    }
}
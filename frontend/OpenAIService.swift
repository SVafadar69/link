import Foundation
import Combine

class OpenAIService: ObservableObject {
    private let apiKey =  ProcessInfo.processInfo.environment["OPENAI_API_KEY"]
    private let apiURL = "https://api.openai.com/v1/chat/completions"
    
    @Published var responseText: String = ""
    
    func sendMessage(_ message: String) {
        guard let url = URL(string: apiURL) else { return }
        
        let messages = [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": message]
        ]
        
        let json: [String: Any] = [
            "model": "gpt-5",
            "messages": messages
        ]
        
        guard let jsonData = try? JSONSerialization.data(withJSONObject: json) else { return }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = jsonData
        
        URLSession.shared.dataTask(with: request) { data, _, error in
            guard let data = data, error == nil else {
                print(error?.localizedDescription ?? "Unknown error")
                return
            }
            
            if let result = try? JSONDecoder().decode(OpenAIResponse.self, from: data) {
                DispatchQueue.main.async {
                    self.responseText = result.choices.first?.message.content ?? "Either got no response or failed to retrieve response."
                }
            }
        }.resume()
    }
}

// MARK: - Response Models
struct OpenAIResponse: Codable {
    let choices: [Choice]
}

struct Choice: Codable {
    let message: Message
}

struct Message: Codable {
    let role: String
    let content: String
}


import SwiftUI
import Combine

struct Messages: View {
    @State private var messages: [String] = ["Let the AI know of anything you want to be notified of in advance (i.e. let me know if grandma falls over)"]
    @State private var inputText: String = ""
    private let openAIService = OpenAIService()

    var body: some View {
        VStack {
            List(messages, id: \.self) { message in
                Text(message)
            }
            HStack {
                TextField("Type a message", text: $inputText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding()
                Button(action: sendMessage) {
                    Text("Send")
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
            }
            .padding()
        }
    }
    
    func sendMessage() {
        let userMessage = inputText
        messages.append("You: \(userMessage)")
        inputText = ""
        
        Task {
            let responseMessage = await openAIService.sendMessage(userMessage)
            await MainActor.run {
                    messages.append("Bot: \(responseMessage)")
                }
            }
            
        }
}

#Preview {
    Messages()
}

import SwiftUI

struct Try: View {
    @StateObject private var aiService = OpenAIService()
    @State private var userInput: String = ""
    
    var body: some View {
        NavigationView {
            VStack {
                ScrollView {
                    Text(aiService.responseText)
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(10)
                        .padding()
                }
                
                HStack {
                    TextField("Ask me something...", text: $userInput)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                    
                    Button("Send") {
                        aiService.sendMessage(userInput)
                        userInput = ""
                    }
                    .padding(.leading, 5)
                }
                .padding()
            }
            .navigationTitle("AI Chat")
        }
    }
}

#Preview {
    Try()
}

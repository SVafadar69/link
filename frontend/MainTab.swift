import SwiftUI

struct MainTab: View {
    var body: some View {
        TabView {
            
            ContentView()
                .tabItem {
                    Label("Video Feed", systemImage: "video.fill")
                }
            AskLLM()
                .tabItem {
                    Label("Ask LLM", systemImage: "door.left.hand.open")
                }
            Messages()
                .tabItem{
                    Label("Guardian", systemImage: "exclamationmark.bubble.fill")
            }
            
            // Stored Events Tab
            Clips()
                .tabItem {
                    Label("Moments", systemImage: "film.stack")
                }
        }
        }
    }



#Preview {
    MainTab()
}

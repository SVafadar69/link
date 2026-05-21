import SwiftUI
import AVKit
import WebKit

struct ContentView: View {
    private let player = AVPlayer(
        url: URL(String://)
    )

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VideoPlayer(player: player)
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .onAppear {
                    player.play()
                }

            VStack(spacing: 0) {
                HStack {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 8, height: 8)

                    Text("LIVE")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(.red)

                    Spacer()
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(Color(white: 0.1).opacity(0.8))
                .zIndex(1)

                Spacer()
            }
        }
    }
}

#Preview {
    ContentView()
}

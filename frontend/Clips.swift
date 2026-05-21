import SwiftUI
import AVKit

struct VideoClips: Identifiable {
    let id = UUID()
    let title: String
    let timestamp: String
    let duration: String
    let thumbnailName: String
}

struct Clips: View {
    let clips: [VideoClips] = [
        VideoClips(
            title: "Person Detected at Front Door",
            timestamp: "Today: 8:42pm",
            duration: "0.18s",
            thumbnailName: "video.fill"
        ),
        VideoClips(
            title: "Motion Detected",
            timestamp: "Today • 6:15 PM",
            duration: "0:11",
            thumbnailName: "video.fill"
        ),
        VideoClips(
            title: "Package Delivered",
            timestamp: "Yesterday • 2:04 PM",
            duration: "0:23",
            thumbnailName: "shippingbox.fill"
        ),
        VideoClips(
            title: "Vehicle Passed By",
            timestamp: "Yesterday • 11:32 AM",
            duration: "0:09",
            thumbnailName: "car.fill"
        )
    ]
    
    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(clips) { clip in
                        VideoClipCard(clip: clip)}
                }
            }
            .padding()
        }
        .background(Color(.systemGroupedBackground))
    }
}


#Preview {
    Clips()
}

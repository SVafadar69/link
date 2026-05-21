import SwiftUI

struct VideoClipCard: View {
    let clip: VideoClips
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            
            ZStack {
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color.black.opacity(0.9))
                    .frame(height: 190)
                
                Image(systemName: clip.thumbnailName)
                    .font(.system(size: 44, weight: .medium))
                    .foregroundColor(.white.opacity(0.9))
                
                VStack {
                    Spacer()
                    
                    HStack {
                        Spacer()
                        
                        Text(clip.duration)
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(Color.black.opacity(0.7))
                            .clipShape(Capsule())
                    }
                    .padding(12)
                }
                
                Circle()
                    .fill(Color.white.opacity(0.9))
                    .frame(width: 54, height: 54)
                    .overlay(
                        Image(systemName: "play.fill")
                            .font(.title3)
                            .foregroundColor(.black)
                            .offset(x: 2)
                    )
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text(clip.title)
                    .font(.headline)
                    .foregroundColor(.primary)
                    .lineLimit(2)
                
                HStack {
                    Image(systemName: "clock")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text(clip.timestamp)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Button {
                        print("More options tapped")
                    } label: {
                        Image(systemName: "ellipsis")
                            .font(.headline)
                            .foregroundColor(.secondary)
                            .padding(8)
                    }
                }
            }
            .padding()
        }
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .shadow(color: Color.black.opacity(0.08), radius: 8, x: 0, y: 4)
    }
}

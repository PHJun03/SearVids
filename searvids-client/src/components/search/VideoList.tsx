/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

import VideoCard from './VideoCard';
import type { VideoWithTags } from '../../types';

interface VideoListProps {
  videos: VideoWithTags[];
  totalCount: number;
}

export default function VideoList({ videos, totalCount }: VideoListProps) {
  if (videos.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500 text-lg">No videos found</p>
      </div>
    );
  }

  return (
    <div>
      <p className="text-gray-600 mb-4">Found {totalCount} video(s)</p>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {videos.map((video) => (
          <VideoCard key={video.id} video={video} />
        ))}
      </div>
    </div>
  );
}
/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

import { Link } from 'react-router-dom';
import { Play } from 'lucide-react';
import { videoApi } from '../../services/api';
import type { VideoWithTags } from '../../types';

interface VideoCardProps {
  video: VideoWithTags;
}

export default function VideoCard({ video }: VideoCardProps) {
  const formatFileSize = (bytes: number) => {
    const gb = (bytes / (1024 * 1024 * 1024)).toFixed(2);
    return `${gb} GB`;
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${secs}s`;
  };

  return (
    <Link
      to={`/video/${video.id}`}
      className="bg-white rounded-lg shadow hover:shadow-lg transition overflow-hidden"
    >
      <div className="relative bg-gray-800 h-40 flex items-center justify-center">
        <img
          src={videoApi.getThumbnailUrl(video.id)}
          alt={video.title}
          className="w-full h-full object-cover"
          onError={(e) => {
            (e.target as HTMLImageElement).src =
              'https://via.placeholder.com/300x200?text=No+Thumbnail';
          }}
        />
        <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-30 flex items-center justify-center transition">
          <Play className="text-white" size={40} />
        </div>
      </div>
      <div className="p-4">
        <h3 className="font-semibold text-lg mb-2 line-clamp-2">{video.title}</h3>
        <div className="text-sm text-gray-600 space-y-1 mb-3">
          <p>Resolution: {video.width}x{video.height}</p>
          <p>Duration: {formatDuration(video.duration)}</p>
          <p>Size: {formatFileSize(video.fileSize)}</p>
        </div>
        {video.tags.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {video.tags.slice(0, 3).map((tag) => (
              <span
                key={tag.id}
                className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded"
              >
                {tag.name}
              </span>
            ))}
            {video.tags.length > 3 && (
              <span className="text-xs text-gray-500">+{video.tags.length - 3}</span>
            )}
          </div>
        )}
      </div>
    </Link>
  );
}
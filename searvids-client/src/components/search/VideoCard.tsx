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
      className="bg-slate-800 rounded-xl shadow-lg hover:shadow-xl hover:bg-slate-750 transition-all overflow-hidden border border-slate-700 group"
    >
      <div className="relative bg-slate-900 aspect-video flex items-center justify-center overflow-hidden">
        <img
          src={videoApi.getThumbnailUrl(video.id, 0)}
          alt={video.title}
          className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
          onError={(e) => {
            (e.target as HTMLImageElement).src =
              'https://via.placeholder.com/300x200?text=No+Thumbnail';
          }}
        />
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 flex items-center justify-center transition-colors duration-300">
          <div className="bg-white/20 backdrop-blur-sm p-3 rounded-full opacity-0 group-hover:opacity-100 transform scale-75 group-hover:scale-100 transition-all duration-300">
            <Play className="text-white fill-white" size={24} />
          </div>
        </div>
        <div className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
          {formatDuration(video.duration)}
        </div>
      </div>
      <div className="p-4">
        <h3 className="font-semibold text-lg mb-2 line-clamp-2 text-slate-100 group-hover:text-blue-400 transition-colors">{video.title}</h3>
        <div className="text-sm text-slate-400 space-y-1 mb-3">
          <div className="flex justify-between">
            <span>{video.width}x{video.height}</span>
            <span>{formatFileSize(video.fileSize)}</span>
          </div>
        </div>
        {video.tags.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {video.tags.slice(0, 3).map((tag) => (
              <span
                key={tag.id}
                className="bg-slate-700 text-slate-300 text-xs px-2 py-1 rounded border border-slate-600"
              >
                {tag.name}
              </span>
            ))}
            {video.tags.length > 3 && (
              <span className="text-xs text-slate-500">+{video.tags.length - 3}</span>
            )}
          </div>
        )}
      </div>
    </Link>
  );
}

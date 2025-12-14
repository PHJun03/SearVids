/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import ReactPlayer from 'react-player';
import { formatDistanceToNow } from 'date-fns';
import Loading from '../components/common/Loading';
import Error from '../components/common/Error';
import { videoApi } from '../services/api';

export default function VideoDetail() {
  const { id } = useParams<{ id: string }>();
  const videoId = id ? parseInt(id, 10) : 0;

  const { data: video, isLoading, error } = useQuery({
    queryKey: ['video', videoId],
    queryFn: () => videoApi.getVideo(videoId),
    enabled: videoId > 0,
  });

  if (!id || Number.isNaN(videoId)) {
    return <Error message="Invalid video ID" />;
  }

  if (isLoading) return <Loading />;

  if (error || !video) {
    return <Error message={(error as Error)?.message || 'Video not found'} />;
  }

  const formatFileSize = (bytes: number) => `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  const formatDuration = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Video Player */}
      <div className="bg-black rounded-xl overflow-hidden mb-8 shadow-2xl border border-slate-800">
        <ReactPlayer
          url={videoApi.getVideoStreamUrl(video.id)}
          controls
          width="100%"
          height="500px"
        />
      </div>

      {/* Video Info */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <h1 className="text-3xl font-bold mb-4 text-slate-100">{video.title}</h1>

          <div className="bg-slate-800 rounded-xl p-6 mb-6 border border-slate-700">
            <h2 className="text-lg font-semibold mb-4 text-slate-200">Details</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-slate-400 text-sm">Resolution</p>
                <p className="font-semibold text-slate-200">
                  {video.width}x{video.height}
                </p>
              </div>
              <div>
                <p className="text-slate-400 text-sm">FPS</p>
                <p className="font-semibold text-slate-200">{video.fps}</p>
              </div>
              <div>
                <p className="text-slate-400 text-sm">Duration</p>
                <p className="font-semibold text-slate-200">{formatDuration(video.duration)}</p>
              </div>
              <div>
                <p className="text-slate-400 text-sm">File Size</p>
                <p className="font-semibold text-slate-200">{formatFileSize(video.fileSize)}</p>
              </div>
              <div>
                <p className="text-slate-400 text-sm">Codec</p>
                <p className="font-semibold text-slate-200">{video.codec}</p>
              </div>
              <div>
                <p className="text-slate-400 text-sm">Bitrate</p>
                <p className="font-semibold text-slate-200">{video.bitrate} kbps</p>
              </div>
            </div>
          </div>

          {/* Tags */}
          {video.tags.length > 0 && (
            <div className="mb-6">
              <h2 className="text-lg font-semibold mb-3 text-slate-200">Tags</h2>
              <div className="flex flex-wrap gap-2">
                {video.tags.map((tag) => (
                  <span
                    key={tag.id}
                    className="bg-blue-500/20 text-blue-300 px-3 py-1 rounded-full border border-blue-500/30"
                  >
                    {tag.name}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="lg:col-span-1">
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-lg font-semibold mb-4 text-slate-200">File Information</h2>
            <div className="space-y-3 text-sm">
              <div>
                <p className="text-slate-400">File Path</p>
                <p className="font-mono text-xs break-all text-slate-300">{video.filePath}</p>
              </div>
              <div>
                <p className="text-slate-400">Added</p>
                <p className="text-slate-300">{formatDistanceToNow(new Date(video.addedAt), { addSuffix: true })}</p>
              </div>
              <div>
                <p className="text-slate-400">Created</p>
                <p className="text-slate-300">{new Date(video.createdAt).toLocaleDateString()}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

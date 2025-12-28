/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

import { useState, useEffect, useMemo, useCallback } from 'react';
import { useMutation, useQuery, useQueryClient, keepPreviousData } from '@tanstack/react-query';
import { Search, Video, ArrowRight, Loader2, Upload, FileVideo, X } from 'lucide-react';
import { videoApi, type AnalyzeResponse, type AnalyzeStatus, type SearchResponse } from '../services/api';
import Error from '../components/common/Error';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'url' | 'file'>('url');
  const [url, setUrl] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [keyword, setKeyword] = useState('');
  const [videoId, setVideoId] = useState<string | null>(null);
  const [analyzedUrl, setAnalyzedUrl] = useState<string | null>(null);
  const [dots, setDots] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const queryClient = useQueryClient();

  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);
    return () => clearInterval(interval);
  }, []);

  // Cleanup local video on refresh/close
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (videoId && videoId.startsWith('local_')) {
        // Use fetch with keepalive for reliable cleanup on unload
        fetch(`/api/videos/${videoId}`, { method: 'DELETE', keepalive: true });
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      // Also cleanup on component unmount if we want strict "session" behavior?
      // No, unmount happens on navigation too.
    };
  }, [videoId]);

  // start analyze url
  const {
    mutate: analyzeUrl,
    isPending: isUrlPending,
    error: urlError,
  } = useMutation<AnalyzeResponse, Error, { url: string; query: string }>({
    mutationFn: videoApi.analyzeVideo,
    onSuccess: (resp) => {
      setVideoId(resp.video_id);
      setAnalyzedUrl(url);
    },
  });

  // upload file
  const {
    mutate: uploadFile,
    isPending: isUploadPending,
    error: uploadError,
  } = useMutation<AnalyzeResponse, Error, File>({
    mutationFn: videoApi.uploadVideo,
    onSuccess: (resp) => {
      setVideoId(resp.video_id);
      setAnalyzedUrl(file?.name || 'Local Video');
    },
  });

  const isPending = isUrlPending || isUploadPending;
  const analyzeError = urlError || uploadError;

  const handleAnalyze = () => {
    queryClient.removeQueries({ queryKey: ['search-chapters'] });
    setVideoId(null);
    
    if (activeTab === 'url' && url) {
      setAnalyzedUrl(url);
      analyzeUrl({ url, query: '' });
    } else if (activeTab === 'file' && file) {
      setAnalyzedUrl(file.name);
      uploadFile(file);
    }
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith('video/')) {
        setFile(droppedFile);
        setActiveTab('file');
      }
    }
  }, []);


  // WebSocket for status updates
  useEffect(() => {
    if (!videoId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/ws/videos/${videoId}/status`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'subscribe', video_id: videoId }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'status') {
          queryClient.setQueryData(['analyze-status', videoId], data);
        }
      } catch (e) {
        console.error('WS parse error', e);
      }
    };

    return () => {
      ws.close();
    };
  }, [videoId, queryClient]);

  // poll status (fallback)
  const {
    data: analyzeStatus,
    error: statusError,
  } = useQuery<AnalyzeStatus>({
    queryKey: ['analyze-status', videoId],
    queryFn: () => videoApi.getAnalyzeStatus(videoId as string),
    enabled: !!videoId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 1000;
      return data.status === 'done' || data.status === 'error' ? false : 2000;
    },
  });



  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const getVideoLink = (videoUrl: string, startTime: number) => {
    const time = Math.floor(startTime);
    if (videoUrl.includes('youtube.com') || videoUrl.includes('youtu.be')) {
      const separator = videoUrl.includes('?') ? '&' : '?';
      return `${videoUrl}${separator}t=${time}`;
    }
    return `${videoUrl}#t=${time}`;
  };

  const renderStatus = () => {
    if (!videoId) return null;
    if (statusError) return <Error message="Failed to fetch status." />;
    if (!analyzeStatus) return <div className="flex items-center justify-center gap-2 text-blue-400"><Loader2 className="animate-spin" /> Waiting for status...</div>;
    
    const isDone = analyzeStatus.status === 'done';
    const isError = analyzeStatus.status === 'error';
    const isAnalyzing = analyzeStatus.status === 'analyzing';

    return (
      <div className="w-full max-w-2xl bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700 space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold text-slate-200">Analysis Status</h3>
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
            isDone ? 'bg-emerald-500/20 text-emerald-400' : 
            isError ? 'bg-red-500/20 text-red-400' : 
            'bg-blue-500/20 text-blue-400'
          }`}>
            {isAnalyzing ? `ANALYZING${dots}` : analyzeStatus.status.toUpperCase()}
          </span>
        </div>

        {isDone && <p className="text-emerald-400 text-sm text-center">Analysis completed successfully!</p>}
        {isError && <p className="text-red-400 text-sm text-center">Error: {analyzeStatus.error}</p>}
      </div>
    );
  };

  // chapters search
  const {
    data: searchData,
    isFetching: isSearching,
    error: searchError,
  } = useQuery<SearchResponse>({
    queryKey: ['search-chapters', keyword, videoId, analyzeStatus?.indexed_visual_frames, analyzeStatus?.indexed_audio_segments],
    queryFn: () => videoApi.searchChapters(keyword, videoId || undefined),
    enabled: !!keyword && !!videoId && !!analyzeStatus && (analyzeStatus.status === 'done' || analyzeStatus.indexed_visual_frames > 0 || analyzeStatus.indexed_audio_segments > 0),
    staleTime: 0,
    placeholderData: keepPreviousData,
  });

  const mergedResults = useMemo(() => {
    const results = searchData?.results ?? [];
    if (!results.length) return [];

    const sortedResults = [...results].sort((a, b) => a.start_time - b.start_time);
    const merged: { start: number; end: number; items: typeof results }[] = [];
    
    if (sortedResults.length > 0) {
      let currentGroup = {
        start: sortedResults[0].start_time,
        end: sortedResults[0].end_time,
        items: [sortedResults[0]]
      };
      
      for (let i = 1; i < sortedResults.length; i++) {
        const item = sortedResults[i];
        if (item.start_time <= currentGroup.end + 3.0) {
          currentGroup.end = Math.max(currentGroup.end, item.end_time);
          currentGroup.items.push(item);
        } else {
          merged.push(currentGroup);
          currentGroup = {
            start: item.start_time,
            end: item.end_time,
            items: [item]
          };
        }
      }
      merged.push(currentGroup);
    }
    return merged;
  }, [searchData]);

  const renderChapters = () => {
    if (isSearching && !searchData) return <div className="flex items-center gap-2 text-blue-400"><Loader2 className="animate-spin" /> Searching chapters...</div>;
    if (searchError) return <Error message="Failed to load chapters." />;
    
    if (!mergedResults.length) {
      // Only show "No results" if analysis is fully done and we still found nothing
      if (analyzeStatus?.status === 'done') {
        return <p className="text-slate-500">No matching moments found.</p>;
      }
      // Otherwise (analyzing, pending, etc.), show nothing to avoid flickering
      return null;
    }

    return (
      <div className="w-full max-w-2xl space-y-4">
        <h3 className="text-xl font-bold text-slate-200 mb-4">Search Results</h3>
        {mergedResults.map((group) => {
          const visualItems = group.items.filter(i => i.caption === 'visual_frame');
          
          const bestVisual = visualItems.sort((a, b) => b.similarity - a.similarity)[0];
          const thumbnailItem = bestVisual || group.items[0];

          return (
            <a 
              key={`group-${group.start}`} 
              href={analyzedUrl ? getVideoLink(analyzedUrl, group.start) : '#'}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-4 bg-slate-800 hover:bg-slate-750 border border-slate-700 p-4 rounded-xl transition-colors block text-decoration-none"
            >
              <div className="relative group cursor-pointer">
                <img
                  className="w-40 aspect-video object-cover rounded-lg bg-slate-900"
                  src={videoApi.getThumbnailUrl(videoId!, thumbnailItem.start_time)}
                  alt="thumbnail"
                />
              </div>
              
              <div className="flex-1 min-w-0">
                <p className="text-lg text-emerald-400 font-mono font-bold">
                  {formatTime(group.start)} - {formatTime(group.end)}
                </p>
              </div>
            </a>
          );
        })}
      </div>
    );
  };

  return (
    <div className="flex flex-col items-center justify-center py-20 px-4">
      {/* Hero Section */}
      <div className="text-center mb-12 max-w-2xl">
        <h1 className="text-5xl md:text-6xl font-extrabold mb-6 tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
          SearVids
        </h1>
        <p className="text-xl text-slate-400 font-light">
          Analyze videos and search for specific moments using AI.
        </p>
      </div>

      {/* Input Section */}
      <div className="w-full max-w-2xl bg-slate-800/50 backdrop-blur-sm p-8 rounded-2xl border border-slate-700 shadow-xl mb-12">
        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setActiveTab('url')}
            className={`flex-1 py-2 text-sm font-medium rounded-lg transition-colors ${
              activeTab === 'url'
                ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}
          >
            Video URL
          </button>
          <button
            onClick={() => setActiveTab('file')}
            className={`flex-1 py-2 text-sm font-medium rounded-lg transition-colors ${
              activeTab === 'file'
                ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}
          >
            Upload File
          </button>
        </div>

        <div className="space-y-6">
          {activeTab === 'url' ? (
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300 ml-1">Video URL</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Video className="text-slate-500" size={20} />
                </div>
                <input
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com/video.mp4"
                  className="w-full pl-12 pr-4 py-3 bg-slate-900/50 border border-slate-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white placeholder-slate-500 transition-all"
                />
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300 ml-1">Local Video File</label>
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                  isDragging
                    ? 'border-blue-500 bg-blue-500/10'
                    : 'border-slate-600 bg-slate-900/30 hover:border-slate-500'
                }`}
              >
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => {
                    if (e.target.files?.[0]) setFile(e.target.files[0]);
                  }}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                {file ? (
                  <div className="flex items-center justify-center gap-3 text-emerald-400">
                    <FileVideo size={32} />
                    <span className="font-medium truncate max-w-[200px]">{file.name}</span>
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        setFile(null);
                      }}
                      className="p-1 hover:bg-slate-700 rounded-full text-slate-400 hover:text-white transition-colors z-10 relative"
                    >
                      <X size={16} />
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2 text-slate-400 pointer-events-none">
                    <Upload className="mx-auto mb-2" size={32} />
                    <p className="font-medium">Click to upload or drag and drop</p>
                    <p className="text-xs text-slate-500">MP4, WebM, MKV (max 2GB)</p>
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-300 ml-1">Search Query</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Search className="text-slate-500" size={20} />
              </div>
              <input
                type="text"
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                placeholder="e.g., 'cat jumping' or 'hello world'"
                className="w-full pl-12 pr-4 py-3 bg-slate-900/50 border border-slate-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white placeholder-slate-500 transition-all"
              />
            </div>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={isPending || (!url && activeTab === 'url') || (!file && activeTab === 'file') || !keyword}
            className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-white py-4 rounded-xl font-bold text-lg transition-all flex items-center justify-center gap-2 shadow-lg hover:shadow-blue-500/20"
          >
            {isPending ? (
              <>
                <Loader2 className="animate-spin" size={20} />
                {activeTab === 'file' ? 'Uploading & Analyzing...' : 'Starting Analysis...'}
              </>
            ) : (
              <>
                Generate Chapters
                <ArrowRight size={20} />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Results Section */}
      <div className="w-full flex flex-col items-center space-y-8">
        {analyzeError && (
          <div className="w-full max-w-2xl">
            <Error message="Failed to start analysis. Please check the URL and try again." />
          </div>
        )}
        {renderStatus()}
        {renderChapters()}
      </div>
    </div>
  );
}

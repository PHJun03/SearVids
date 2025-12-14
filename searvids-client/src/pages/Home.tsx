/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

import { useState, useEffect, useMemo } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Search, Video, ArrowRight, Loader2 } from 'lucide-react';
import { videoApi, type AnalyzeResponse, type AnalyzeStatus, type SearchResponse } from '../services/api';
import Error from '../components/common/Error';

export default function Home() {
  const [url, setUrl] = useState(() => sessionStorage.getItem('searvids_url') || '');
  const [keyword, setKeyword] = useState(() => sessionStorage.getItem('searvids_keyword') || '');
  const [videoId, setVideoId] = useState<string | null>(() => sessionStorage.getItem('searvids_videoId'));
  const [dots, setDots] = useState('');
  const queryClient = useQueryClient();

  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);
    return () => clearInterval(interval);
  }, []);

  // Persist state to sessionStorage
  useEffect(() => {
    sessionStorage.setItem('searvids_url', url);
  }, [url]);

  useEffect(() => {
    sessionStorage.setItem('searvids_keyword', keyword);
  }, [keyword]);

  useEffect(() => {
    if (videoId) {
      sessionStorage.setItem('searvids_videoId', videoId);
    } else {
      sessionStorage.removeItem('searvids_videoId');
    }
  }, [videoId]);

  // start analyze
  const {
    mutate,
    isPending,
    error: analyzeError,
  } = useMutation<AnalyzeResponse, Error, { url: string; query: string }>({
    mutationFn: videoApi.analyzeVideo,
    onSuccess: (resp) => setVideoId(resp.video_id),
  });

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

  const handleAnalyze = (e: React.FormEvent) => {
    e.preventDefault();
    if (url && keyword) {
      setVideoId(null);
      mutate({ url, query: keyword });
    }
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
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
      if (isSearching) return null;
      if (analyzeStatus?.status === 'pending' || !analyzeStatus) return null;
      return <p className="text-slate-500">No results found yet.</p>;
    }

    return (
      <div className="w-full max-w-2xl space-y-4">
        <h3 className="text-xl font-bold text-slate-200 mb-4">Search Results</h3>
        {mergedResults.map((group) => {
          const visualItems = group.items.filter(i => i.caption === 'visual_frame');
          
          const bestVisual = visualItems.sort((a, b) => b.similarity - a.similarity)[0];
          const thumbnailItem = bestVisual || group.items[0];

          return (
            <div key={`group-${group.start}`} className="flex items-center gap-4 bg-slate-800 hover:bg-slate-750 border border-slate-700 p-4 rounded-xl transition-colors">
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
            </div>
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
          Searvids
        </h1>
        <p className="text-xl text-slate-400 font-light">
          Analyze videos and search for specific moments using AI.
        </p>
      </div>

      {/* Input Section */}
      <div className="w-full max-w-2xl bg-slate-800/50 backdrop-blur-sm p-8 rounded-2xl border border-slate-700 shadow-xl mb-12">
        <form onSubmit={handleAnalyze} className="space-y-6">
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
            type="submit"
            disabled={isPending || !url || !keyword}
            className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-white py-4 rounded-xl font-bold text-lg transition-all flex items-center justify-center gap-2 shadow-lg hover:shadow-blue-500/20"
          >
            {isPending ? (
              <>
                <Loader2 className="animate-spin" size={20} />
                Starting Analysis...
              </>
            ) : (
              <>
                Generate Chapters
                <ArrowRight size={20} />
              </>
            )}
          </button>
        </form>
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

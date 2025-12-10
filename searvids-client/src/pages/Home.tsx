/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Search, Video, ArrowRight } from 'lucide-react';
import { videoApi, type AnalyzeResponse, type AnalyzeStatus, type SearchResponse } from '../services/api';
import Loading from '../components/common/Loading';
import Error from '../components/common/Error';

export default function Home() {
  const [url, setUrl] = useState('');
  const [keyword, setKeyword] = useState('');
  const [videoId, setVideoId] = useState<string | null>(null);

  // start analyze
  const {
    mutate,
    isPending,
    error: analyzeError,
  } = useMutation<AnalyzeResponse, Error, { url: string; query: string }>({
    mutationFn: videoApi.analyzeVideo,
    onSuccess: (resp) => setVideoId(resp.video_id),
  });

  // poll status
  const {
    data: analyzeStatus,
    isFetching: isPolling,
    error: statusError,
  } = useQuery<AnalyzeStatus>({
    queryKey: ['analyze-status', videoId],
    queryFn: () => videoApi.getAnalyzeStatus(videoId as string),
    enabled: !!videoId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 2000;
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
    if (!analyzeStatus) return <p className="text-blue-200">Waiting for status...</p>;
    return (
      <div className="text-center py-4 space-y-2">
        <p className="text-lg font-semibold">
          Status: <span className="text-emerald-300">{analyzeStatus.status}</span>
        </p>
        {analyzeStatus.current_stage && (
          <p className="text-sm text-gray-300">Stage: {analyzeStatus.current_stage}</p>
        )}
        <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden max-w-md mx-auto">
          <div
            className="bg-blue-500 h-2 transition-all"
            style={{ width: `${Math.min(100, analyzeStatus.progress_percent)}%` }}
          />
        </div>
        {analyzeStatus.status === 'done' && (
          <p className="text-emerald-300 text-sm">Analysis completed.</p>
        )}
        {analyzeStatus.status === 'error' && (
          <p className="text-red-300 text-sm">Error: {analyzeStatus.error}</p>
        )}
      </div>
    );
  };

  // chapters search
  const {
    data: searchData,
    isFetching: isSearching,
    error: searchError,
  } = useQuery<SearchResponse>({
    queryKey: ['search-chapters', keyword, analyzeStatus?.indexed_visual_frames, analyzeStatus?.indexed_audio_segments],
    queryFn: () => videoApi.searchChapters(keyword),
    enabled: !!keyword && !!analyzeStatus && (analyzeStatus.status === 'done' || analyzeStatus.indexed_visual_frames > 0 || analyzeStatus.indexed_audio_segments > 0),
    staleTime: 0,
  });

  const renderChapters = () => {
    if (isSearching && !searchData) return <p className="text-blue-200">Searching chapters...</p>;
    if (searchError) return <Error message="Failed to load chapters." />;
    const results = searchData?.results ?? [];
    
    if (!results.length) {
      if (isSearching) return <p className="text-blue-200">Searching chapters...</p>;
      // Only show "No results" if we are not searching and have no results, 
      // but also check if we have started analysis.
      if (analyzeStatus?.status === 'pending' || !analyzeStatus) return null;
      
      return (
        <div className="text-center space-y-2">
          <p className="text-gray-400">No results found yet.</p>
        </div>
      );
    }

    // Sort by start time
    const sortedResults = [...results].sort((a, b) => a.start_time - b.start_time);

    // Merge overlapping results
    const mergedResults: { start: number; end: number; items: typeof results }[] = [];
    if (sortedResults.length > 0) {
      let currentGroup = {
        start: sortedResults[0].start_time,
        end: sortedResults[0].end_time,
        items: [sortedResults[0]]
      };
      
      for (let i = 1; i < sortedResults.length; i++) {
        const item = sortedResults[i];
        // Check overlap (using < end_time for simple overlap)
        if (item.start_time < currentGroup.end) {
          currentGroup.end = Math.max(currentGroup.end, item.end_time);
          currentGroup.items.push(item);
        } else {
          mergedResults.push(currentGroup);
          currentGroup = {
            start: item.start_time,
            end: item.end_time,
            items: [item]
          };
        }
      }
      mergedResults.push(currentGroup);
    }

    return (
      <div className="w-full max-w-2xl space-y-4">
        {mergedResults.map((group, idx) => {
          // Determine display properties
          const audioItem = group.items.find(i => i.caption !== 'visual_frame');
          const visualItems = group.items.filter(i => i.caption === 'visual_frame');
          
          // Use audio caption if available, otherwise "Visual Match"
          const caption = audioItem ? audioItem.caption : 'Visual Match';
          
          // Find best visual item for thumbnail, or fallback to first item
          const bestVisual = visualItems.sort((a, b) => b.similarity - a.similarity)[0];
          const thumbnailId = bestVisual ? bestVisual.id : group.items[0].id;
          
          const maxScore = Math.max(...group.items.map(i => i.similarity));
          const hasAudio = !!audioItem;
          const hasVisual = visualItems.length > 0;

          return (
            <div key={`group-${idx}`} className="flex gap-3 items-center bg-gray-800/70 p-3 rounded-xl">
              <img
                className="w-32 h-20 object-cover rounded"
                src={videoApi.getThumbnailUrl(thumbnailId)}
                alt="thumbnail"
              />
              <div className="flex-1">
                <div className="flex justify-between items-start">
                  <p className="text-sm text-emerald-200">
                    {formatTime(group.start)} - {formatTime(group.end)}
                  </p>
                  <div className="flex gap-1">
                    {hasAudio && <span className="text-[10px] bg-blue-900 text-blue-200 px-1.5 py-0.5 rounded">Audio</span>}
                    {hasVisual && <span className="text-[10px] bg-purple-900 text-purple-200 px-1.5 py-0.5 rounded">Visual</span>}
                  </div>
                </div>
                <p className="text-base font-semibold text-white line-clamp-2">{caption}</p>
                <p className="text-xs text-gray-400">score: {maxScore.toFixed(3)}</p>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white flex items-center justify-center">
      <div className="max-w-4xl w-full mx-auto px-4 py-20 flex flex-col items-center">

        {/* Header Section */}
        <div className="flex flex-col items-center text-center mb-16 mx-auto">
          <h1 className="text-6xl font-extrabold mb-6 tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
            Searvids
          </h1>
          <p className="text-xl text-gray-300 font-light">
            Search in a video, generate video chapters about the keyword.
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-white/10 backdrop-blur-lg p-8 rounded-2xl shadow-2xl border border-white/10 mb-12 w-full max-w-2xl mx-auto flex flex-col items-center">
          <form onSubmit={handleAnalyze} className="w-full space-y-4 text-center">
            {/* Video URL Input */}
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Video className="text-red-500" size={24} />
              </div>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="Paste Video URL here..."
                className="w-full pl-12 pr-12 py-4 bg-gray-800/50 border border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white placeholder-gray-400 transition-all text-center"
              />
            </div>

            {/* Keyword Input & Button */}
            <div className="flex gap-4 flex-col md:flex-row md:items-center md:justify-center">
              <div className="relative flex-1 md:max-w-md w-full">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Search className="text-blue-400" size={20} />
                </div>
                <input
                  type="text"
                  value={keyword}
                  onChange={(e) => setKeyword(e.target.value)}
                  placeholder="Search Keyword..."
                  className="w-full pl-12 pr-12 py-4 bg-gray-800/50 border border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white placeholder-gray-400 transition-all text-center"
                />
              </div>
              <button
                type="submit"
                disabled={isPending || !url || !keyword}
                className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-8 py-4 rounded-xl font-bold text-lg transition-all flex items-center justify-center gap-2 shadow-lg hover:shadow-blue-500/30 w-full md:w-auto"
              >
                {isPending ? 'Analyzing...' : 'Generate Chapters'}
                {!isPending && <ArrowRight size={20} />}
              </button>
            </div>
          </form>
        </div>

       {/* Results / Status Section */}
        <div className="space-y-6 w-full max-w-2xl mx-auto flex flex-col items-center">
          {(isPending || isPolling) && (
            <div className="text-center py-12">
              <Loading />
              <p className="mt-4 text-blue-200 animate-pulse">Analyzing video content...</p>
            </div>
          )}
          {analyzeError && (
            <div className="text-center w-full">
              <Error message="Failed to start analysis. Please check the URL and try again." />
            </div>
          )}
          {renderStatus()}
          {renderChapters()}
        </div>
      </div>
    </div>
  );
}
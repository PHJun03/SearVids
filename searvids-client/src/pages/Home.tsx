import { useState, useEffect } from 'react';
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
      searchMutation.reset();
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
  const searchMutation = useMutation<SearchResponse, Error, { query: string }>({
    mutationFn: ({ query }) => videoApi.searchChapters(query),
  });

  // trigger search when analysis done
  useEffect(() => {
    if (analyzeStatus?.status === 'done' && keyword && !searchMutation.isPending && !searchMutation.isSuccess) {
      searchMutation.mutate({ query: keyword });
    }
  }, [analyzeStatus?.status, keyword, searchMutation.isPending, searchMutation.isSuccess]);

  const renderChapters = () => {
    if (searchMutation.isPending) return <p className="text-blue-200">Searching chapters...</p>;
    if (searchMutation.error) return <Error message="Failed to load chapters." />;
    const results = searchMutation.data?.results ?? [];
    if (!results.length) return null;
    return (
      <div className="w-full max-w-2xl space-y-4">
        {results.map((r) => (
          <div key={`${r.id}-${r.start_time}`} className="flex gap-3 items-center bg-gray-800/70 p-3 rounded-xl">
            <img
              className="w-32 h-20 object-cover rounded"
              src={videoApi.getThumbnailUrl(r.id)}
              alt="thumbnail"
            />
            <div className="flex-1">
              <p className="text-sm text-emerald-200">
                {formatTime(r.start_time)} - {formatTime(r.end_time)}
              </p>
              <p className="text-base font-semibold text-white line-clamp-2">{r.caption || 'No caption'}</p>
              <p className="text-xs text-gray-400">score: {r.similarity.toFixed(3)}</p>
            </div>
          </div>
        ))}
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
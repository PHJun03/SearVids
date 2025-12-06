import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Search, Video, Clock, ArrowRight } from 'lucide-react';
import { videoApi } from '../services/api';
import Loading from '../components/common/Loading';
import Error from '../components/common/Error';

export default function Home() {
  const [url, setUrl] = useState('');
  const [topic, setTopic] = useState('');

  const { mutate, isPending, error, data: chapters } = useMutation({
    mutationFn: videoApi.analyzeVideo,
  });

  const handleAnalyze = (e: React.FormEvent) => {
    e.preventDefault();
    if (url && topic) {
      mutate({ url, query: topic });
    }
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
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
            Search in a video, generate video chapters about the topic you searched.
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

            {/* Topic Input & Button */}
            <div className="flex gap-4 flex-col md:flex-row md:items-center md:justify-center">
              <div className="relative flex-1 md:max-w-md w-full">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Search className="text-blue-400" size={20} />
                </div>
                <input
                  type="text"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder="Search Topic..."
                  className="w-full pl-12 pr-12 py-4 bg-gray-800/50 border border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white placeholder-gray-400 transition-all text-center"
                />
              </div>
              <button
                type="submit"
                disabled={isPending || !url || !topic}
                className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-8 py-4 rounded-xl font-bold text-lg transition-all flex items-center justify-center gap-2 shadow-lg hover:shadow-blue-500/30 w-full md:w-auto"
              >
                {isPending ? 'Analyzing...' : 'Generate Chapters'}
                {!isPending && <ArrowRight size={20} />}
              </button>
            </div>
          </form>
        </div>

        {/* Results Section */}
        <div className="space-y-6 w-full max-w-2xl mx-auto flex flex-col items-center">
          {isPending && (
            <div className="text-center py-12">
              <Loading />
              <p className="mt-4 text-blue-200 animate-pulse">Analyzing video content...</p>
            </div>
          )}

          {error && (
            <div className="text-center w-full">
              <Error message="Failed to generate video chapters. Please check the URL and try again." />
            </div>
          )}

          {chapters && chapters.length > 0 && (
            <div className="animate-fade-in-up w-full">
              <h2 className="text-2xl font-bold mb-6 flex items-center justify-center gap-2">
                <Clock className="text-emerald-400" />
                Generated Chapters
              </h2>
              <div className="grid gap-4">
                {chapters.map((chapter) => (
                  <div
                    key={chapter.id}
                    className="bg-gray-800/80 hover:bg-gray-700/80 border border-gray-700 rounded-xl p-4 flex gap-4 transition-all cursor-pointer group"
                    onClick={() => window.open(`${url}&t=${chapter.timestamp}s`, '_blank')}
                  >
                    {/* Thumbnail */}
                    <div className="relative w-40 h-24 flex-shrink-0 rounded-lg overflow-hidden bg-black mx-auto md:mx-0">
                      <img
                        src={chapter.thumbnailUrl}
                        alt={`Chapter at ${formatTime(chapter.timestamp)}`}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                      <div className="absolute bottom-1 right-1 bg-black/80 text-white text-xs px-1.5 py-0.5 rounded">
                        {formatTime(chapter.timestamp)}
                      </div>
                    </div>

                    {/* Content */}
                    <div className="flex-1 flex flex-col justify-center">
                      <h3 className="text-lg font-semibold text-blue-300 mb-1 group-hover:text-blue-200">
                        {formatTime(chapter.timestamp)} - {chapter.description}
                      </h3>
                      <p className="text-gray-400 text-sm line-clamp-2">
                        Click to watch this segment in the video.
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
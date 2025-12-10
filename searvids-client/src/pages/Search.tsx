import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import SearchBar from '../components/search/SearchBar';
import VideoList from '../components/search/VideoList';
import Pagination from '../components/search/Pagination';
import Loading from '../components/common/Loading';
import Error from '../components/common/Error';
import { videoApi } from '../services/api';

export default function Search() {
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(1);
  const pageSize = 12;

  const { data, isLoading, error } = useQuery({
    queryKey: ['videos', query, page],
    queryFn: () =>
      videoApi.searchVideos({
        query: query || undefined,
        page,
        pageSize,
      }),
    enabled: query.length > 0,
    retry: false,
    refetchOnWindowFocus: false,
  });

  const handleSearch = (searchQuery: string) => {
    setQuery(searchQuery);
    setPage(1);
  };

  const handlePageChange = (newPage: number) => {
    setPage(newPage);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Search Videos</h1>

      <SearchBar onSearch={handleSearch} loading={isLoading} />

      {isLoading && <Loading />}
      {error && <Error message={(error as Error).message} />}
      {data && (
        <>
          <VideoList videos={data.items} totalCount={data.totalCount} />
          {data.totalPages > 1 && (
            <Pagination
              currentPage={page}
              totalPages={data.totalPages}
              onPageChange={handlePageChange}
            />
          )}
        </>
      )}
    </div>
  );
}
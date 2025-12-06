import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

/*
// Pages
import Home from './pages/Home';
import Search from './pages/Search';
import VideoDetail from './pages/VideoDetail';
*/

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-gray-100">
          <Routes>
            <Route path="/" element={<div>Home Page (Coming Soon)</div>} />
            <Route path="/search" element={<div>Search Page (Coming Soon)</div>} />
            <Route path="/video/:id" element={<div>Video Detail (Coming Soon)</div>} />
          </Routes>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
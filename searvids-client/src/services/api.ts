import axios from 'axios';
import type { VideoWithTags, VideoTag, SearchParams, PaginatedResponse } from '../types';

const API_BASE_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export const videoApi = {
  searchVideos: async (params: SearchParams) => {
    const response = await api.get<PaginatedResponse<VideoWithTags>>('/videos/search', {
      params,
    });
    return response.data;
  },

  getVideo: async (id: number) => {
    const response = await api.get<VideoWithTags>(`/videos/${id}`);
    return response.data;
  },

  getVideoStreamUrl: (id: number) => {
    return `${API_BASE_URL}/videos/${id}/stream`;
  },

  getThumbnailUrl: (id: number) => {
    return `${API_BASE_URL}/videos/${id}/thumbnail`;
  },

  getAllTags: async () => {
    const response = await api.get<VideoTag[]>('/tags');
    return response.data;
  },
};

export default api;
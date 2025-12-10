/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

import axios from 'axios';
import type { VideoWithTags, VideoTag, SearchParams, PaginatedResponse, AnalyzeRequest } from '../types';

const API_BASE_URL = "/api";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export type AnalyzeResponse = { status: string; video_id: string };
export type AnalyzeStatus = {
  status: 'pending' | 'analyzing' | 'done' | 'error';
  error?: string;
  analyzing: boolean;
  done: boolean;
  progress_percent: number;
  current_stage: string;
  duration_ms: number;
  nb_frames: number;
  indexed_visual_frames: number;
  indexed_audio_segments: number;
};

export type SearchResult = {
  id: string;
  start_time: number;
  end_time: number;
  caption: string;
  similarity: number;
};

export type SearchResponse = { status: string; count?: number; results: SearchResult[] };

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

  getThumbnailUrl: (id: number | string) => `${API_BASE_URL}/videos/${id}/thumbnail`,

  getAllTags: async () => {
    const response = await api.get<VideoTag[]>('/tags');
    return response.data;
  },

  analyzeVideo: async (params: AnalyzeRequest) => {
    const response = await api.post<AnalyzeResponse>('/analyze', params);
    return response.data;
  },

  getAnalyzeStatus: async (videoId: string) => {
    const response = await api.get<AnalyzeStatus>(`/videos/${videoId}/status`);
    return response.data;
  },

  searchChapters: async (query: string, topk = 10): Promise<SearchResponse> => {
    const res = await api.post<SearchResponse>('/search', { query, topk });
    return res.data;
  },
};

export default api;
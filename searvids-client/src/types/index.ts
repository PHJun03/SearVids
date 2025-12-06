export interface VideoMetadata {
  id: number;
  title: string;
  filePath: string;
  fileSize: number;
  duration: number;
  width: number;
  height: number;
  fps: number;
  codec: string;
  bitrate: number;
  createdAt: string;
  addedAt: string;
  thumbnailPath?: string;
}

export interface VideoTag {
  id: number;
  name: string;
}

export interface VideoWithTags extends VideoMetadata {
  tags: VideoTag[];
}

export interface SearchParams {
  query?: string;
  tags?: string[];
  dateFrom?: string;
  dateTo?: string;
  sort?: 'date' | 'title';
  page?: number;
  pageSize?: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  totalCount: number;
  page: number;
  pageSize: number;
  totalPages: number;
}
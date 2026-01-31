/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

#include "ffmpeg_decoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace ffmpeg_decoder {

static std::atomic<bool> g_ffmpeg_initialized{false};

void init_ffmpeg() {
  bool expected = false;
  if (g_ffmpeg_initialized.compare_exchange_strong(expected, true)) {
    av_log_set_level(AV_LOG_WARNING); // Show warnings/errors
    avformat_network_init();
  }
}

inline int64_t av_rescale_q_int64(int64_t a, AVRational src_tb,
                                  AVRational dst_tb) {
  return av_rescale_q(a, src_tb, dst_tb);
}

int64_t avtime_to_ms(int64_t avtime, int64_t time_base_num,
                     int64_t time_base_den) {
  if (time_base_den == 0)
    return 0;
  double t_seconds =
      double(avtime) * double(time_base_num) / double(time_base_den);
  return static_cast<int64_t>(t_seconds * 1000.0 + 0.5);
}

static void throw_av_err(int errcode, const std::string &ctx) {
  char buf[256];
  av_strerror(errcode, buf, sizeof(buf));
  std::ostringstream oss;
  oss << ctx << ": " << buf << " (err=" << errcode << ")";
  throw std::runtime_error(oss.str());
}

static AVFormatContext *open_input_format_context(const std::string &path) {
  AVFormatContext *fmt = nullptr;
  AVDictionary *opts = nullptr;
  int ret = avformat_open_input(&fmt, path.c_str(), nullptr, &opts);
  if (ret < 0 || !fmt) {
    throw_av_err(ret, "Failed to open input: " + path);
  }
  ret = avformat_find_stream_info(fmt, nullptr);
  if (ret < 0) {
    avformat_close_input(&fmt);
    throw_av_err(ret, "Failed to find stream info: " + path);
  }
  return fmt;
}

static AVCodecContext *open_decoder_for_stream(AVStream *stream) {
  const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
  if (!codec) {
    throw std::runtime_error("Decoder not found for codec id: " +
                             std::to_string(stream->codecpar->codec_id));
  }
  AVCodecContext *cctx = avcodec_alloc_context3(codec);
  if (!cctx)
    throw std::runtime_error("Failed to allocate codec context");
  int ret = avcodec_parameters_to_context(cctx, stream->codecpar);
  if (ret < 0) {
    avcodec_free_context(&cctx);
    throw_av_err(ret, "Failed to copy codec parameters to context");
  }
  ret = avcodec_open2(cctx, codec, nullptr);
  if (ret < 0) {
    avcodec_free_context(&cctx);
    throw_av_err(ret, "Failed to open codec");
  }
  return cctx;
}

VideoInfo probe(const std::string &path) {
  init_ffmpeg();

  auto fmt_deleter = [](AVFormatContext *ctx) {
    if (ctx)
      avformat_close_input(&ctx);
  };
  std::unique_ptr<AVFormatContext, decltype(fmt_deleter)> fmt(nullptr,
                                                              fmt_deleter);
  fmt.reset(open_input_format_context(path));

  VideoInfo info;
  info.format_name =
      fmt->iformat ? std::string(fmt->iformat->name ? fmt->iformat->name : "")
                   : "";

  if (fmt->duration != AV_NOPTS_VALUE) {
    info.duration_ms =
        static_cast<int64_t>(fmt->duration / (AV_TIME_BASE / 1000));
  } else {
    info.duration_ms = 0;
  }

  for (unsigned i = 0; i < fmt->nb_streams; ++i) {
    AVStream *st = fmt->streams[i];
    if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && !info.has_video) {
      info.has_video = true;
      info.width = st->codecpar->width;
      info.height = st->codecpar->height;

      double fps = 0.0;
      if (st->avg_frame_rate.num && st->avg_frame_rate.den)
        fps = av_q2d(st->avg_frame_rate);
      if (fps <= 0.0 && st->r_frame_rate.num && st->r_frame_rate.den)
        fps = av_q2d(st->r_frame_rate);
      info.fps = fps;

      if (st->nb_frames > 0)
        info.nb_frames = st->nb_frames;

      if (st->codecpar->codec_id != AV_CODEC_ID_NONE) {
        const AVCodec *codec = avcodec_find_decoder(st->codecpar->codec_id);
        if (codec && codec->name)
          info.codec_name = codec->name;
      }
    } else if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO &&
               !info.has_audio) {
      info.has_audio = true;
    }
  }

  return info;
}

int count_frames(const std::string &path) {
  init_ffmpeg();

  auto fmt_deleter = [](AVFormatContext *ctx) {
    if (ctx)
      avformat_close_input(&ctx);
  };
  std::unique_ptr<AVFormatContext, decltype(fmt_deleter)> fmt_guard(
      nullptr, fmt_deleter);
  fmt_guard.reset(open_input_format_context(path));

  int video_stream_index = av_find_best_stream(
      fmt_guard.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (video_stream_index < 0)
    return 0;

  AVStream *vstream = fmt_guard->streams[video_stream_index];

  if (vstream->nb_frames > 0) {
    if (vstream->nb_frames > INT32_MAX)
      return INT32_MAX;
    return static_cast<int>(vstream->nb_frames);
  }

  auto codec_context_deleter = [](AVCodecContext *ctx) {
    if (ctx)
      avcodec_free_context(&ctx);
  };
  std::unique_ptr<AVCodecContext, decltype(codec_context_deleter)> codec_ctx(
      nullptr, codec_context_deleter);
  codec_ctx.reset(open_decoder_for_stream(vstream));

  AVPacket *pkt = av_packet_alloc();
  if (!pkt)
    throw std::runtime_error("Failed to allocate AVPacket");
  AVFrame *frame = av_frame_alloc();
  if (!frame) {
    av_packet_free(&pkt);
    throw std::runtime_error("Failed to allocate AVFrame");
  }

  int frame_count = 0;
  int ret = 0;

  av_seek_frame(fmt_guard.get(), video_stream_index, 0, AVSEEK_FLAG_BACKWARD);

  while ((ret = av_read_frame(fmt_guard.get(), pkt)) >= 0) {
    if (pkt->stream_index == video_stream_index) {
      ret = avcodec_send_packet(codec_ctx.get(), pkt);
      if (ret < 0) {
        av_packet_unref(pkt);
        continue;
      }
      while (true) {
        ret = avcodec_receive_frame(codec_ctx.get(), frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
          break;
        if (ret < 0)
          break;
        ++frame_count;
        av_frame_unref(frame);
        if (frame_count == INT32_MAX)
          break;
      }
    }
    av_packet_unref(pkt);
  }

  avcodec_send_packet(codec_ctx.get(), nullptr);
  while (true) {
    ret = avcodec_receive_frame(codec_ctx.get(), frame);
    if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
      break;
    if (ret < 0)
      break;
    ++frame_count;
    av_frame_unref(frame);
    if (frame_count == INT32_MAX)
      break;
  }

  av_frame_free(&frame);
  av_packet_free(&pkt);

  if (frame_count < 0)
    frame_count = 0;
  if (frame_count > INT32_MAX)
    frame_count = INT32_MAX;
  return static_cast<int>(frame_count);
}

bool extract_audio(const std::string &video_path, const std::string &audio_path,
                   int sample_rate, int channels,
                   std::function<void(double)> progress_callback) {
  init_ffmpeg();

  // 1) Open input video file
  auto fmt_deleter = [](AVFormatContext *ctx) {
    if (ctx)
      avformat_close_input(&ctx);
  };
  std::unique_ptr<AVFormatContext, decltype(fmt_deleter)> fmt_ctx(nullptr,
                                                                  fmt_deleter);

  try {
    fmt_ctx.reset(open_input_format_context(video_path));
  } catch (const std::exception &e) {
    std::cerr << "Failed to open video: " << e.what() << std::endl;
    return false;
  }

  // 2) Find audio stream
  int audio_stream_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_AUDIO,
                                             -1, -1, nullptr, 0);
  if (audio_stream_idx < 0) {
    std::cerr << "No audio stream found in: " << video_path << std::endl;
    return false;
  }

  AVStream *audio_stream = fmt_ctx->streams[audio_stream_idx];

  // 3) Open audio decoder
  auto codec_deleter = [](AVCodecContext *ctx) {
    if (ctx)
      avcodec_free_context(&ctx);
  };
  std::unique_ptr<AVCodecContext, decltype(codec_deleter)> codec_ctx(
      nullptr, codec_deleter);

  try {
    codec_ctx.reset(open_decoder_for_stream(audio_stream));
  } catch (const std::exception &e) {
    std::cerr << "Failed to open audio decoder: " << e.what() << std::endl;
    return false;
  }

  // ...existing code...
  // 4) Setup SwrContext for resampling
  SwrContext *swr_ctx = swr_alloc();
  if (!swr_ctx) {
    std::cerr << "Failed to allocate SwrContext" << std::endl;
    return false;
  }

#if LIBAVUTIL_VERSION_MAJOR >= 57
  AVChannelLayout in_ch_layout = codec_ctx->ch_layout;
  AVChannelLayout out_ch_layout;
  av_channel_layout_default(&out_ch_layout, channels);

  av_opt_set_chlayout(swr_ctx, "in_chlayout", &in_ch_layout, 0);
  av_opt_set_int(swr_ctx, "in_sample_rate", codec_ctx->sample_rate, 0);
  av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", codec_ctx->sample_fmt, 0);

  av_opt_set_chlayout(swr_ctx, "out_chlayout", &out_ch_layout, 0);
#else
  // FFmpeg < 5 compatibility (uses uint64 channel_layout)
  uint64_t in_ch_layout =
      codec_ctx->channel_layout
          ? codec_ctx->channel_layout
          : av_get_default_channel_layout(codec_ctx->channels);
  uint64_t out_ch_layout =
      (channels == 1) ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO;

  av_opt_set_int(swr_ctx, "in_channel_layout", in_ch_layout, 0);
  av_opt_set_int(swr_ctx, "in_sample_rate", codec_ctx->sample_rate, 0);
  av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", codec_ctx->sample_fmt, 0);

  av_opt_set_int(swr_ctx, "out_channel_layout", out_ch_layout, 0);
#endif

  av_opt_set_int(swr_ctx, "out_sample_rate", sample_rate, 0);
  av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_S16,
                        0); // 16-bit PCM

  if (swr_init(swr_ctx) < 0) {
    std::cerr << "Failed to initialize SwrContext" << std::endl;
    swr_free(&swr_ctx);
    return false;
  }

  // 5) Open output WAV file
  FILE *out_file = fopen(audio_path.c_str(), "wb");
  if (!out_file) {
    std::cerr << "Failed to open output file: " << audio_path << std::endl;
    swr_free(&swr_ctx);
    return false;
  }

  // Write WAV header (44 bytes placeholder, will update later)
  uint8_t wav_header[44] = {0};
  fwrite(wav_header, 1, 44, out_file);

  // 6) Decode and resample audio
  AVPacket *pkt = av_packet_alloc();
  AVFrame *frame = av_frame_alloc();
  if (!pkt || !frame) {
    std::cerr << "Failed to allocate packet/frame" << std::endl;
    if (pkt)
      av_packet_free(&pkt);
    if (frame)
      av_frame_free(&frame);
    swr_free(&swr_ctx);
    fclose(out_file);
    return false;
  }

  int64_t total_samples = 0;
  std::vector<uint8_t> resample_buffer;

  while (av_read_frame(fmt_ctx.get(), pkt) >= 0) {
    if (pkt->stream_index == audio_stream_idx) {
      int ret = avcodec_send_packet(codec_ctx.get(), pkt);
      if (ret < 0) {
        av_packet_unref(pkt);
        continue;
      }

      while (ret >= 0) {
        ret = avcodec_receive_frame(codec_ctx.get(), frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
          break;
        if (ret < 0)
          break;

        // Calculate output sample count
        int out_samples = av_rescale_rnd(
            swr_get_delay(swr_ctx, codec_ctx->sample_rate) + frame->nb_samples,
            sample_rate, codec_ctx->sample_rate, AV_ROUND_UP);

        // Allocate output buffer
        int out_buffer_size = av_samples_get_buffer_size(
            nullptr, channels, out_samples, AV_SAMPLE_FMT_S16, 1);
        resample_buffer.resize(out_buffer_size);

        uint8_t *out_ptr = resample_buffer.data();
        int converted =
            swr_convert(swr_ctx, &out_ptr, out_samples,
                        (const uint8_t **)frame->data, frame->nb_samples);

        if (converted > 0) {
          int bytes_to_write = converted * channels * sizeof(int16_t);
          fwrite(resample_buffer.data(), 1, bytes_to_write, out_file);
          total_samples += converted;
        }

        av_frame_unref(frame);
      }
    }
    av_packet_unref(pkt);
  }

  // Flush decoder
  avcodec_send_packet(codec_ctx.get(), nullptr);
  int ret;
  while ((ret = avcodec_receive_frame(codec_ctx.get(), frame)) >= 0) {
    int out_samples = av_rescale_rnd(
        swr_get_delay(swr_ctx, codec_ctx->sample_rate) + frame->nb_samples,
        sample_rate, codec_ctx->sample_rate, AV_ROUND_UP);

    int out_buffer_size = av_samples_get_buffer_size(
        nullptr, channels, out_samples, AV_SAMPLE_FMT_S16, 1);
    resample_buffer.resize(out_buffer_size);

    uint8_t *out_ptr = resample_buffer.data();
    int converted =
        swr_convert(swr_ctx, &out_ptr, out_samples,
                    (const uint8_t **)frame->data, frame->nb_samples);

    if (converted > 0) {
      int bytes_to_write = converted * channels * sizeof(int16_t);
      fwrite(resample_buffer.data(), 1, bytes_to_write, out_file);
      total_samples += converted;
    }

    av_frame_unref(frame);
  }

  // Flush resampler
  while (true) {
    int out_samples = sample_rate; // 1 second buffer
    int out_buffer_size = av_samples_get_buffer_size(
        nullptr, channels, out_samples, AV_SAMPLE_FMT_S16, 1);
    resample_buffer.resize(out_buffer_size);

    uint8_t *out_ptr = resample_buffer.data();
    int converted = swr_convert(swr_ctx, &out_ptr, out_samples, nullptr, 0);

    if (converted <= 0)
      break;

    int bytes_to_write = converted * channels * sizeof(int16_t);
    fwrite(resample_buffer.data(), 1, bytes_to_write, out_file);
    total_samples += converted;
  }

  // 7) Update WAV header
  int32_t data_size = total_samples * channels * sizeof(int16_t);
  int32_t file_size = data_size + 36; // 44 - 8

  fseek(out_file, 0, SEEK_SET);

  // RIFF header
  fwrite("RIFF", 1, 4, out_file);
  fwrite(&file_size, 4, 1, out_file);
  fwrite("WAVE", 1, 4, out_file);

  // fmt subchunk
  fwrite("fmt ", 1, 4, out_file);
  int32_t fmt_chunk_size = 16;
  fwrite(&fmt_chunk_size, 4, 1, out_file);
  int16_t audio_format = 1; // PCM
  fwrite(&audio_format, 2, 1, out_file);
  int16_t num_channels = channels;
  fwrite(&num_channels, 2, 1, out_file);
  int32_t sample_rate_val = sample_rate;
  fwrite(&sample_rate_val, 4, 1, out_file);
  int32_t byte_rate = sample_rate * channels * sizeof(int16_t);
  fwrite(&byte_rate, 4, 1, out_file);
  int16_t block_align = channels * sizeof(int16_t);
  fwrite(&block_align, 2, 1, out_file);
  int16_t bits_per_sample = 16;
  fwrite(&bits_per_sample, 2, 1, out_file);

  // data subchunk
  fwrite("data", 1, 4, out_file);
  fwrite(&data_size, 4, 1, out_file);

  // Cleanup
  fclose(out_file);
  av_frame_free(&frame);
  av_packet_free(&pkt);
  swr_free(&swr_ctx);

  std::cout << "Audio extracted: " << total_samples << " samples, "
            << (total_samples / sample_rate) << " seconds" << std::endl;

  return true;
}

int64_t ms_to_avtime(int64_t ms, int64_t time_base_num, int64_t time_base_den) {
  if (time_base_num == 0)
    return 0;
  double t_seconds = static_cast<double>(ms) / 1000.0;
  return static_cast<int64_t>((t_seconds * time_base_den) / time_base_num +
                              0.5);
}

FrameData extract_frame_at(const std::string &video_path, int64_t timestamp_ms,
                           bool seek_backward, int target_width,
                           int target_height) {
  init_ffmpeg();

  auto fmt_deleter = [](AVFormatContext *ctx) {
    if (ctx)
      avformat_close_input(&ctx);
  };
  std::unique_ptr<AVFormatContext, decltype(fmt_deleter)> fmt_ctx(nullptr,
                                                                  fmt_deleter);

  try {
    fmt_ctx.reset(open_input_format_context(video_path));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to open video: " + std::string(e.what()));
  }

  // Find video stream
  int video_stream_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_VIDEO,
                                             -1, -1, nullptr, 0);
  if (video_stream_idx < 0) {
    throw std::runtime_error("No video stream found");
  }

  AVStream *video_stream = fmt_ctx->streams[video_stream_idx];

  // Open video decoder
  auto codec_deleter = [](AVCodecContext *ctx) {
    if (ctx)
      avcodec_free_context(&ctx);
  };
  std::unique_ptr<AVCodecContext, decltype(codec_deleter)> codec_ctx(
      nullptr, codec_deleter);

  try {
    codec_ctx.reset(open_decoder_for_stream(video_stream));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to open decoder: " +
                             std::string(e.what()));
  }

  // Seek to timestamp
  int64_t seek_target = av_rescale_q(timestamp_ms, {1, 1000}, // milliseconds
                                     video_stream->time_base);

  int seek_flags = seek_backward ? AVSEEK_FLAG_BACKWARD : 0;
  if (av_seek_frame(fmt_ctx.get(), video_stream_idx, seek_target, seek_flags) <
      0) {
    std::cerr << "Warning: Seek failed, reading from start" << std::endl;
  }

  avcodec_flush_buffers(codec_ctx.get());

  // Read frames until we get the target frame
  AVPacket *pkt = av_packet_alloc();
  AVFrame *frame = av_frame_alloc();
  if (!pkt || !frame) {
    if (pkt)
      av_packet_free(&pkt);
    if (frame)
      av_frame_free(&frame);
    throw std::runtime_error("Failed to allocate packet/frame");
  }

  FrameData result;
  bool frame_found = false;
  int ret = 0;

  while (av_read_frame(fmt_ctx.get(), pkt) >= 0) {
    if (pkt->stream_index == video_stream_idx) {
      ret = avcodec_send_packet(codec_ctx.get(), pkt);
      if (ret < 0) {
        av_packet_unref(pkt);
        continue;
      }

      while (ret >= 0) {
        ret = avcodec_receive_frame(codec_ctx.get(), frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
          break;
        if (ret < 0)
          break;

        // Get frame timestamp
        int64_t frame_pts = frame->pts;
        int64_t frame_ms =
            av_rescale_q(frame_pts, video_stream->time_base, {1, 1000});

        // Check if this is the frame we want
        if (frame_ms >= timestamp_ms || !seek_backward) {
          // Determine output dimensions
          int dst_width = (target_width > 0) ? target_width : frame->width;
          int dst_height = (target_height > 0) ? target_height : frame->height;

          // Convert to RGB24 using swscale
          SwsContext *sws_ctx = sws_getContext(
              frame->width, frame->height,
              static_cast<AVPixelFormat>(frame->format), dst_width, dst_height,
              AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);

          if (!sws_ctx) {
            av_frame_unref(frame);
            throw std::runtime_error("Failed to create SwsContext");
          }

          // Allocate RGB buffer
          int rgb_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, dst_width,
                                                  dst_height, 1);
          result.rgb_data.resize(rgb_size);

          uint8_t *rgb_ptrs[4] = {result.rgb_data.data(), nullptr, nullptr,
                                  nullptr};
          int rgb_linesizes[4] = {dst_width * 3, 0, 0, 0};

          // Convert frame to RGB24
          sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                    rgb_ptrs, rgb_linesizes);

          sws_freeContext(sws_ctx);

          result.width = dst_width;
          result.height = dst_height;
          result.timestamp_ms = frame_ms;

          frame_found = true;
          av_frame_unref(frame);
          break;
        }

        av_frame_unref(frame);
      }

      if (frame_found)
        break;
    }
    av_packet_unref(pkt);
  }

  av_frame_free(&frame);
  av_packet_free(&pkt);

  if (!frame_found) {
    throw std::runtime_error("Failed to extract frame at timestamp " +
                             std::to_string(timestamp_ms) + "ms");
  }

  return result;
}

void extract_frames_with_callback(
    const std::string &video_path,
    std::function<void(const FrameData &)> callback, double interval_seconds,
    int max_frames, int64_t start_time_ms, int64_t end_time_ms,
    int target_width, int target_height, FrameExtractionMethod method,
    double min_scene_len) {
  init_ffmpeg();

  // Get video info first
  VideoInfo info = probe(video_path);
  if (!info.has_video) {
    throw std::runtime_error("No video stream found in: " + video_path);
  }

  // Determine time range
  int64_t video_duration_ms = info.duration_ms;
  if (end_time_ms == 0 || end_time_ms > video_duration_ms) {
    end_time_ms = video_duration_ms;
  }

  if (start_time_ms >= end_time_ms) {
    throw std::runtime_error("Invalid time range: start >= end");
  }

  // Calculate frame timestamps if using INTERVAL
  std::vector<int64_t> timestamps;
  if (method == FrameExtractionMethod::INTERVAL) {
    int64_t interval_ms = static_cast<int64_t>(interval_seconds * 1000.0);

    for (int64_t t = start_time_ms; t < end_time_ms; t += interval_ms) {
      timestamps.push_back(t);
      if (max_frames > 0 &&
          timestamps.size() >= static_cast<size_t>(max_frames)) {
        break;
      }
    }

    std::cout << "Extracting " << timestamps.size() << " frames from "
              << video_path << " (interval: " << interval_seconds << "s)"
              << std::endl;
  } else if (method == FrameExtractionMethod::KEYFRAMES) {
    std::cout << "Extracting keyframes from " << video_path << std::endl;
  } else {
    std::cout << "Extracting scene-change frames from " << video_path
              << std::endl;
  }

  // Open format context
  auto fmt_deleter = [](AVFormatContext *ctx) {
    if (ctx)
      avformat_close_input(&ctx);
  };
  std::unique_ptr<AVFormatContext, decltype(fmt_deleter)> fmt_ctx(nullptr,
                                                                  fmt_deleter);

  try {
    fmt_ctx.reset(open_input_format_context(video_path));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to open video: " + std::string(e.what()));
  }

  // Find video stream
  int video_stream_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_VIDEO,
                                             -1, -1, nullptr, 0);
  if (video_stream_idx < 0) {
    throw std::runtime_error("No video stream found");
  }

  AVStream *video_stream = fmt_ctx->streams[video_stream_idx];

  // Open decoder
  auto codec_deleter = [](AVCodecContext *ctx) {
    if (ctx)
      avcodec_free_context(&ctx);
  };
  std::unique_ptr<AVCodecContext, decltype(codec_deleter)> codec_ctx(
      nullptr, codec_deleter);

  try {
    codec_ctx.reset(open_decoder_for_stream(video_stream));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to open decoder: " +
                             std::string(e.what()));
  }

  // Determine output dimensions
  int dst_width = (target_width > 0) ? target_width : codec_ctx->width;
  int dst_height = (target_height > 0) ? target_height : codec_ctx->height;

  // Setup SwsContext for RGB conversion and resizing
  SwsContext *sws_ctx = sws_getContext(
      codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt, dst_width,
      dst_height, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);

  if (!sws_ctx) {
    throw std::runtime_error("Failed to create SwsContext");
  }

  AVPacket *pkt = av_packet_alloc();
  AVFrame *frame = av_frame_alloc();
  if (!pkt || !frame) {
    if (pkt)
      av_packet_free(&pkt);
    if (frame)
      av_frame_free(&frame);
    sws_freeContext(sws_ctx);
    throw std::runtime_error("Failed to allocate packet/frame");
  }

  size_t current_target_idx = 0;
  int ret = 0;
  int extracted_count = 0;

  // Setup Scene Detection Filter if needed
  AVFilterGraph *filter_graph = nullptr;
  AVFilterContext *buffersrc_ctx = nullptr;
  AVFilterContext *buffersink_ctx = nullptr;

  if (method == FrameExtractionMethod::SCENE_DETECT) {
    filter_graph = avfilter_graph_alloc();
    const AVFilter *buffersrc = avfilter_get_by_name("buffer");
    const AVFilter *buffersink = avfilter_get_by_name("buffersink");

    char args[512];
    snprintf(args, sizeof(args),
             "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
             codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
             video_stream->time_base.num, video_stream->time_base.den,
             codec_ctx->sample_aspect_ratio.num,
             codec_ctx->sample_aspect_ratio.den);

    if (avfilter_graph_create_filter(&buffersrc_ctx, buffersrc, "in", args,
                                     nullptr, filter_graph) < 0) {
      throw std::runtime_error("Failed to create buffer source");
    }
    if (avfilter_graph_create_filter(&buffersink_ctx, buffersink, "out",
                                     nullptr, nullptr, filter_graph) < 0) {
      throw std::runtime_error("Failed to create buffer sink");
    }

    // Scene detection filter: select frames where scene change score > 0.3
    // Adaptive: OR if time since last selected > min_scene_len
    AVFilterInOut *outputs = avfilter_inout_alloc();
    AVFilterInOut *inputs = avfilter_inout_alloc();
    outputs->name = av_strdup("in");
    outputs->filter_ctx = buffersrc_ctx;
    outputs->pad_idx = 0;
    outputs->next = nullptr;
    inputs->name = av_strdup("out");
    inputs->filter_ctx = buffersink_ctx;
    inputs->pad_idx = 0;
    inputs->next = nullptr;

    // Construct the select expression
    // eq(n,0): Explicitly select the first frame
    std::string select_expr = "gt(scene,0.1)+eq(n,0)";

    if (min_scene_len > 0.0) {
      // Adaptive logic:
      // gt(t-prev_selected_t, min_scene_len): Force select if too much time
      // passed
      char buf[128];
      snprintf(buf, sizeof(buf), "+gt(t-prev_selected_t,%f)", min_scene_len);
      select_expr += buf;
    }

    // Wrap in select='...'
    std::string filter_spec = "select='" + select_expr + "'";

    if (avfilter_graph_parse_ptr(filter_graph, filter_spec.c_str(), &inputs,
                                 &outputs, nullptr) < 0) {
      throw std::runtime_error("Failed to parse filter graph: " + filter_spec);
    }
    if (avfilter_graph_config(filter_graph, nullptr) < 0) {
      throw std::runtime_error("Failed to config filter graph");
    }
    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);
  }

  while (av_read_frame(fmt_ctx.get(), pkt) >= 0) {
    if (method == FrameExtractionMethod::INTERVAL &&
        current_target_idx >= timestamps.size())
      break;
    if (max_frames > 0 && extracted_count >= max_frames)
      break;

    if (pkt->stream_index == video_stream_idx) {
      // Optimization: If using keyframes, skip non-keyframe packets
      if (method == FrameExtractionMethod::KEYFRAMES &&
          !(pkt->flags & AV_PKT_FLAG_KEY)) {
        av_packet_unref(pkt);
        continue;
      }

      ret = avcodec_send_packet(codec_ctx.get(), pkt);
      if (ret < 0) {
        av_packet_unref(pkt);
        continue;
      }

      while (ret >= 0) {
        ret = avcodec_receive_frame(codec_ctx.get(), frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
          break;
        if (ret < 0)
          break;

        // Get frame timestamp
        int64_t frame_pts = frame->pts;
        int64_t frame_ms =
            av_rescale_q(frame_pts, video_stream->time_base, {1, 1000});

        bool extract_this = false;

        if (method == FrameExtractionMethod::KEYFRAMES) {
          if (frame_ms >= start_time_ms &&
              (end_time_ms == 0 || frame_ms < end_time_ms)) {
            if (frame->key_frame) {
              extract_this = true;
            }
          }
        } else if (method == FrameExtractionMethod::INTERVAL) {
          // Check if this frame matches our target timestamp
          while (current_target_idx < timestamps.size() &&
                 frame_ms >= timestamps[current_target_idx]) {
            extract_this = true;
            current_target_idx++;
          }
        } else if (method == FrameExtractionMethod::SCENE_DETECT) {
          if (frame_ms >= start_time_ms &&
              (end_time_ms == 0 || frame_ms < end_time_ms)) {
            // Feed frame to filter graph
            int src_ret = av_buffersrc_add_frame_flags(
                buffersrc_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
            if (src_ret < 0) {
              char errbuf[128];
              av_strerror(src_ret, errbuf, sizeof(errbuf));
              std::cerr << "buffersrc add failed: " << errbuf << " (" << src_ret
                        << ")" << std::endl;
            } else {
              std::cerr << "Added frame PTS: " << frame->pts << std::endl;
              AVFrame *filt_frame = av_frame_alloc();
              while (true) {
                int f_ret = av_buffersink_get_frame(buffersink_ctx, filt_frame);
                if (f_ret == AVERROR(EAGAIN) || f_ret == AVERROR_EOF)
                  break;
                if (f_ret < 0) {
                  char errbuf[128];
                  av_strerror(f_ret, errbuf, sizeof(errbuf));
                  std::cerr << "buffersink get failed: " << errbuf << " ("
                            << f_ret << ")" << std::endl;
                  break;
                }

                // If we got a frame here, it passed the select filter
                extract_this = true;
                av_frame_unref(filt_frame);
                break;
              }
              av_frame_free(&filt_frame);
            }
          }
        }

        if (extract_this) {
          // Convert to RGB24 and resize
          int rgb_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, dst_width,
                                                  dst_height, 1);

          FrameData frame_data;
          frame_data.rgb_data.resize(rgb_size);
          frame_data.width = dst_width;
          frame_data.height = dst_height;
          frame_data.timestamp_ms = frame_ms;

          uint8_t *rgb_ptrs[4] = {frame_data.rgb_data.data(), nullptr, nullptr,
                                  nullptr};
          int rgb_linesizes[4] = {dst_width * 3, 0, 0, 0};

          sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                    rgb_ptrs, rgb_linesizes);

          if (callback) {
            callback(frame_data);
          }

          if (method == FrameExtractionMethod::KEYFRAMES) {
            std::cout << "Extracted keyframe " << (extracted_count + 1)
                      << " at " << frame_ms << "ms" << std::endl;
          } else if (method == FrameExtractionMethod::SCENE_DETECT) {
            std::cout << "Extracted scene frame " << (extracted_count + 1)
                      << " at " << frame_ms << "ms" << std::endl;
          } else {
            std::cout << "Extracted frame "
                      << current_target_idx // already incremented
                      << "/" << timestamps.size() << " at " << frame_ms << "ms"
                      << std::endl;
          }

          extracted_count++;
        }

        av_frame_unref(frame);
      }
    }
    av_packet_unref(pkt);
  }

  // FLUSH logic
  if (method == FrameExtractionMethod::SCENE_DETECT) {
    std::cerr << "Flushing filter graph..." << std::endl;
    if (av_buffersrc_add_frame_flags(buffersrc_ctx, nullptr, 0) >= 0) {
      AVFrame *filt_frame = av_frame_alloc();
      while (true) {
        int f_ret = av_buffersink_get_frame(buffersink_ctx, filt_frame);
        if (f_ret == AVERROR(EAGAIN) || f_ret == AVERROR_EOF)
          break;
        if (f_ret < 0)
          break;

        // Convert flushed frame
        int dst_width = (target_width > 0) ? target_width : codec_ctx->width;
        int dst_height =
            (target_height > 0) ? target_height : codec_ctx->height;
        int rgb_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, dst_width,
                                                dst_height, 1);

        FrameData frame_data;
        frame_data.rgb_data.resize(rgb_size);
        frame_data.width = dst_width;
        frame_data.height = dst_height;
        int64_t f_pts = filt_frame->pts;
        frame_data.timestamp_ms = av_rescale_q(
            f_pts, buffersink_ctx->inputs[0]->time_base, {1, 1000});

        uint8_t *rgb_ptrs[4] = {frame_data.rgb_data.data(), nullptr, nullptr,
                                nullptr};
        int rgb_linesizes[4] = {dst_width * 3, 0, 0, 0};
        sws_scale(sws_ctx, filt_frame->data, filt_frame->linesize, 0,
                  filt_frame->height, rgb_ptrs, rgb_linesizes);

        if (callback)
          callback(frame_data);
        std::cerr << "Extracted flushed frame at " << frame_data.timestamp_ms
                  << "ms" << std::endl;
        extracted_count++;

        av_frame_unref(filt_frame);
      }
      av_frame_free(&filt_frame);
    }
  }

  if (filter_graph) {
    avfilter_graph_free(&filter_graph);
  }

  // Cleanup
  av_frame_free(&frame);
  av_packet_free(&pkt);
  sws_freeContext(sws_ctx);
}

std::vector<FrameData>
extract_frames(const std::string &video_path, double interval_seconds,
               int max_frames, int64_t start_time_ms, int64_t end_time_ms,
               int target_width, int target_height,
               FrameExtractionMethod method, double min_scene_len) {
  std::vector<FrameData> results;
  extract_frames_with_callback(
      video_path,
      [&results](const FrameData &frame) { results.push_back(frame); },
      interval_seconds, max_frames, start_time_ms, end_time_ms, target_width,
      target_height, method, min_scene_len);
  return results;
}

FFmpegVersion get_ffmpeg_version() {
  FFmpegVersion version;

  unsigned int avformat_ver = avformat_version();
  unsigned int avcodec_ver = avcodec_version();
  unsigned int avutil_ver = avutil_version();
  unsigned int swresample_ver = swresample_version();
  unsigned int swscale_ver = swscale_version();

  auto format_version = [](unsigned int ver) -> std::string {
    int major = (ver >> 16) & 0xFF;
    int minor = (ver >> 8) & 0xFF;
    int micro = ver & 0xFF;
    return std::to_string(major) + "." + std::to_string(minor) + "." +
           std::to_string(micro);
  };

  version.avformat_version = format_version(avformat_ver);
  version.avcodec_version = format_version(avcodec_ver);
  version.avutil_version = format_version(avutil_ver);
  version.swresample_version = format_version(swresample_ver);
  version.swscale_version = format_version(swscale_ver);

  return version;
}

} // namespace ffmpeg_decoder
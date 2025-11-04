#include "ffmpeg_decoder.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/time.h>
#include <libavutil/avutil.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
}

#include <memory>
#include <sstream>
#include <iostream>
#include <vector>
#include <atomic>

namespace ffmpeg_decoder {

static std::atomic<bool> g_ffmpeg_initialized{false};

void init_ffmpeg() {
    // idempotent init
    bool expected = false;
    if (g_ffmpeg_initialized.compare_exchange_strong(expected, true)) {
        av_log_set_level(AV_LOG_ERROR); // reduce noisy logs; adjust if needed
        avformat_network_init();
        // av_register_all() deprecated in modern FFmpeg; avformat library init done by avformat_network_init / implicit
    }
}

inline int64_t av_rescale_q_int64(int64_t a, AVRational src_tb, AVRational dst_tb) {
    return av_rescale_q(a, src_tb, dst_tb);
}

int64_t avtime_to_ms(int64_t avtime, int64_t time_base_num, int64_t time_base_den) {
    // avtime is in stream time_base units: convert to milliseconds
    if (time_base_den == 0) return 0;
    double t_seconds = double(avtime) * double(time_base_num) / double(time_base_den);
    return static_cast<int64_t>(t_seconds * 1000.0 + 0.5);
}

static void throw_av_err(int errcode, const std::string& ctx) {
    char buf[256];
    av_strerror(errcode, buf, sizeof(buf));
    std::ostringstream oss;
    oss << ctx << ": " << buf << " (err=" << errcode << ")";
    throw std::runtime_error(oss.str());
}

static AVFormatContext* open_input_format_context(const std::string& path) {
    AVFormatContext* fmt = nullptr;
    AVDictionary* opts = nullptr;
    // timeout or max probes can be set in opts if desired (e.g. av_dict_set)
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

static AVCodecContext* open_decoder_for_stream(AVStream* stream) {
    const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) {
        throw std::runtime_error("Decoder not found for codec id: " + std::to_string(stream->codecpar->codec_id));
    }
    AVCodecContext* cctx = avcodec_alloc_context3(codec);
    if (!cctx) throw std::runtime_error("Failed to allocate codec context");
    int ret = avcodec_parameters_to_context(cctx, stream->codecpar);
    if (ret < 0) {
        avcodec_free_context(&cctx);
        throw_av_err(ret, "Failed to copy codec parameters to context");
    }

    // Open decoder
    ret = avcodec_open2(cctx, codec, nullptr);
    if (ret < 0) {
        avcodec_free_context(&cctx);
        throw_av_err(ret, "Failed to open codec");
    }
    return cctx;
}

VideoInfo probe(const std::string& path) {
    init_ffmpeg();

    std::unique_ptr<AVFormatContext, decltype(&avformat_close_input)> fmt{nullptr, &avformat_close_input};
    fmt.reset(open_input_format_context(path));

    VideoInfo info;
    info.format_name = fmt->iformat ? std::string(fmt->iformat->name ? fmt->iformat->name : "") : "";

    // duration in ms
    if (fmt->duration != AV_NOPTS_VALUE) {
        info.duration_ms = static_cast<int64_t>(fmt->duration / (AV_TIME_BASE / 1000));
    } else {
        info.duration_ms = 0;
    }

    for (unsigned i = 0; i < fmt->nb_streams; ++i) {
        AVStream* st = fmt->streams[i];
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && !info.has_video) {
            info.has_video = true;
            info.width = st->codecpar->width;
            info.height = st->codecpar->height;
            // fps: try to compute using avg_frame_rate / r_frame_rate
            double fps = 0.0;
            if (st->avg_frame_rate.num && st->avg_frame_rate.den)
                fps = av_q2d(st->avg_frame_rate);
            if (fps <= 0.0 && st->r_frame_rate.num && st->r_frame_rate.den)
                fps = av_q2d(st->r_frame_rate);
            info.fps = fps;
            // try nb_frames if provided, else 0
            if (st->nb_frames > 0) info.nb_frames = st->nb_frames;
            if (st->codecpar->codec_id != AV_CODEC_ID_NONE) {
                const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
                if (codec && codec->name) info.codec_name = codec->name;
            }
        } else if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && !info.has_audio) {
            info.has_audio = true;
        }
    }

    return info;
}

int count_frames(const std::string& path) {
    init_ffmpeg();

    // Open input
    AVFormatContext* fmt = nullptr;
    fmt = open_input_format_context(path);
    // unique_ptr to ensure close
    std::unique_ptr<AVFormatContext, decltype(&avformat_close_input)> fmt_guard{fmt, &avformat_close_input};

    // Find best video stream
    int video_stream_index = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_index < 0) {
        // No video; return 0
        return 0;
    }
    AVStream* vstream = fmt->streams[video_stream_index];

    // If nb_frames field is reliable, return it
    if (vstream->nb_frames > 0) {
        // nb_frames is in stream timebase units; but it's a count so return directly
        if (vstream->nb_frames > INT32_MAX) return INT32_MAX;
        return static_cast<int>(vstream->nb_frames);
    }

    // Open decoder
    std::unique_ptr<AVCodecContext, decltype(&avcodec_free_context)> codec_ctx{nullptr, &avcodec_free_context};
    codec_ctx.reset(open_decoder_for_stream(vstream));

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) throw std::runtime_error("Failed to allocate AVPacket");

    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        av_packet_free(&pkt);
        throw std::runtime_error("Failed to allocate AVFrame");
    }

    int frame_count = 0;
    int ret = 0;

    // Seek to start to ensure deterministic reading
    av_seek_frame(fmt, video_stream_index, 0, AVSEEK_FLAG_BACKWARD);

    // Read packets and decode
    while ((ret = av_read_frame(fmt, pkt)) >= 0) {
        if (pkt->stream_index == video_stream_index) {
            ret = avcodec_send_packet(codec_ctx.get(), pkt);
            if (ret < 0) {
                // ignore decoding errors for robustness but log if needed
                av_packet_unref(pkt);
                continue;
            }
            while (true) {
                ret = avcodec_receive_frame(codec_ctx.get(), frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) {
                    // decoding error: break inner loop
                    break;
                }
                ++frame_count;
                av_frame_unref(frame);
                // optional: early exit if frame_count over some huge threshold
                if (frame_count == INT32_MAX) break;
            }
        }
        av_packet_unref(pkt);
    }

    // flush decoder
    avcodec_send_packet(codec_ctx.get(), nullptr);
    while (true) {
        ret = avcodec_receive_frame(codec_ctx.get(), frame);
        if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) break;
        if (ret < 0) break;
        ++frame_count;
        av_frame_unref(frame);
        if (frame_count == INT32_MAX) break;
    }

    av_frame_free(&frame);
    av_packet_free(&pkt);

    if (frame_count < 0) frame_count = 0;
    if (frame_count > INT32_MAX) frame_count = INT32_MAX;
    return static_cast<int>(frame_count);
}

} // namespace ffmpeg_decoder

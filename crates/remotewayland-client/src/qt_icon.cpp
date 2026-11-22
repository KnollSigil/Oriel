#include <QBuffer>
#include <QByteArray>
#include <QImage>
#include <QImageReader>
#include <QPainter>
#include <QSize>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
constexpr size_t kMaxIconBytes = 4 * 1024 * 1024;
}

extern "C" bool rw_qt_decode_icon_argb8888(
    const uint8_t *data,
    size_t len,
    uint32_t target_size,
    uint8_t *out_buf,
    size_t out_len)
{
    if (!data || len == 0 || len > kMaxIconBytes
        || len > static_cast<size_t>(std::numeric_limits<int>::max())
        || target_size == 0 || target_size > 512 || !out_buf) {
        return false;
    }
    const size_t expected =
        static_cast<size_t>(target_size) * static_cast<size_t>(target_size) * 4;
    if (out_len < expected) {
        return false;
    }

    try {
        QByteArray bytes = QByteArray::fromRawData(
            reinterpret_cast<const char *>(data),
            static_cast<qsizetype>(len));
        QBuffer buffer(&bytes);
        if (!buffer.open(QIODevice::ReadOnly)) {
            return false;
        }

        QImageReader reader(&buffer);
        reader.setAutoTransform(true);
        const QSize orig_size = reader.size();
        if (orig_size.isValid() && !orig_size.isEmpty()) {
            const QSize scaled = orig_size.scaled(
                static_cast<int>(target_size),
                static_cast<int>(target_size),
                Qt::KeepAspectRatio);
            if (!scaled.isEmpty()) {
                reader.setScaledSize(scaled);
            }
        }

        QImage img = reader.read();
        if (img.isNull()) {
            return false;
        }

        const int dim = static_cast<int>(target_size);
        if (img.width() > dim || img.height() > dim || (img.width() != dim && img.height() != dim)) {
            img = img.scaled(dim, dim, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        }

        // Center into a square target_size x target_size premultiplied ARGB8888 buffer
        // as required by xdg_toplevel_icon_v1 (wl_shm::Format::Argb8888).
        QImage square(dim, dim, QImage::Format_ARGB32_Premultiplied);
        square.fill(Qt::transparent);
        {
            QPainter painter(&square);
            painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
            const int x = (dim - img.width()) / 2;
            const int y = (dim - img.height()) / 2;
            painter.drawImage(x, y, img);
        }

        const size_t row_bytes = static_cast<size_t>(target_size) * 4;
        for (int row = 0; row < square.height(); ++row) {
            std::memcpy(
                out_buf + static_cast<size_t>(row) * row_bytes,
                square.constScanLine(row),
                row_bytes);
        }
        return true;
    } catch (...) {
        return false;
    }
}

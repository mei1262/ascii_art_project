import os
import html
from PIL import Image, ImageEnhance, ImageOps

# 70 级细腻字符集
ASCII_CHARS = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "input")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "output")

def process_image(image_path):
    """预处理图片：兼容 PNG/JPG、填充透明背景、增强对比度"""
    img = Image.open(image_path)

    # 处理 PNG 透明通道，转换为纯白底
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert("RGB")

    # 1. 自动对比度拉伸（去灰）
    img = ImageOps.autocontrast(img, cutoff=1)
    
    # 2. 手动强化对比度
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # 3. 转为灰度图
    return img.convert("L")


def save_single_html(ascii_text, output_html_path, title_name, dark_mode=True):
    """为单张图片生成独立的 HTML 网页"""
    bg_color = "#000000" if dark_mode else "#ffffff"
    text_color = "#ffffff" if dark_mode else "#000000"
    safe_ascii_text = html.escape(ascii_text)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>{title_name} - ASCII Art</title>
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
            font-family: "Courier New", Consolas, "Liberation Mono", "Menlo", monospace;
            font-size: 7px;
            line-height: 0.8;
            letter-spacing: 0px;
            white-space: pre;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            margin: 0;
            user-select: all;
        }}
    </style>
</head>
<body>{safe_ascii_text}</body>
</html>"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def save_batch_html(
    art_dict,
    output_html_path="output/all_ascii_art.html",
    dark_mode=True,
    font_size=7,
    line_height=0.8,
    font_family='"Courier New", Consolas, "Liberation Mono", "Menlo", monospace',
):
    """将所有图片的字符画集合在一个带侧边栏菜单的网页中，方便一次性预览"""
    bg_color = "#121212" if dark_mode else "#f5f5f5"
    card_bg = "#000000" if dark_mode else "#ffffff"
    text_color = "#ffffff" if dark_mode else "#000000"
    sidebar_bg = "#1e1e1e" if dark_mode else "#e0e0e0"

    sections = []
    nav_links = []
    for idx, (filename, ascii_text) in enumerate(art_dict.items()):
        safe_text = html.escape(ascii_text)
        sec_id = f"art-{idx}"
        nav_links.append(f'<a href="#{sec_id}">{filename}</a>')
        sections.append(f"""
        <section id="{sec_id}" class="art-card">
            <h2>📷 {filename}</h2>
            <div class="ascii-content">{safe_text}</div>
        </section>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>ASCII Art 图集画廊</title>
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
            font-family: system-ui, -apple-system, sans-serif;
            margin: 0;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}
        .sidebar {{
            width: 240px;
            background-color: {sidebar_bg};
            padding: 20px;
            box-sizing: border-box;
            overflow-y: auto;
            border-right: 1px solid #333;
        }}
        .sidebar h3 {{ margin-top: 0; font-size: 16px; }}
        .sidebar a {{
            display: block;
            color: {text_color};
            text-decoration: none;
            padding: 8px 10px;
            margin-bottom: 5px;
            border-radius: 4px;
            font-size: 14px;
            word-break: break-all;
        }}
        .sidebar a:hover {{
            background-color: rgba(255, 255, 255, 0.1);
        }}
        .main-content {{
            flex: 1;
            padding: 40px;
            overflow-y: auto;
            box-sizing: border-box;
        }}
        .art-card {{
            background-color: {card_bg};
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 40px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .art-card h2 {{
            margin-top: 0;
            font-size: 18px;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }}
        .ascii-content {{
            font-family: {font_family};
            font-size: {font_size}px;
            line-height: {line_height};
            letter-spacing: 0px;
            white-space: pre;
            overflow-x: auto;
            user-select: all;
        }}
    </style>
</head>
<body>
    <div class="sidebar">
        <h3>🖼️ 图片列表 ({len(art_dict)})</h3>
        {''.join(nav_links)}
    </div>
    <div class="main-content">
        {''.join(sections)}
    </div>
</body>
</html>"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def convert_image_to_ascii_str(image_path, output_width=150, font_aspect_ratio=0.75, dark_mode=True):
    """将单张图片转为 ASCII 字符文本"""
    img = process_image(image_path)

    # 自动适应图片宽高比
    img_width, img_height = img.size
    img_aspect_ratio = img_height / img_width
    output_height = int(output_width * img_aspect_ratio * font_aspect_ratio)
    output_height = max(1, output_height)

    img = img.resize((output_width, output_height), Image.Resampling.LANCZOS)

    chars = ASCII_CHARS[::-1] if dark_mode else ASCII_CHARS
    pixels = img.getdata()
    scale = 255 / (len(chars) - 1)
    ascii_str = "".join([chars[int(p / scale)] for p in pixels])

    ascii_lines = [
        ascii_str[i : i + output_width]
        for i in range(0, len(ascii_str), output_width)
    ]
    return "\n".join(ascii_lines), img_width, img_height, output_height


def batch_process_folder(
    input_folder=DEFAULT_INPUT,
    output_folder=DEFAULT_OUTPUT,
    output_width=150,
    font_aspect_ratio=0.75,
    dark_mode=True
):
    """批量处理 input 文件夹内的所有 jpg 和 png 图片"""
    if not os.path.exists(input_folder):
        print(f"❌ 错误：找不到输入文件夹 '{input_folder}'，请创建该文件夹并放入图片！")
        return

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 支持的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        print(f"⚠️ 提示：在 '{input_folder}' 文件夹中没有找到 JPG 或 PNG 图片！")
        return

    print(f"🚀 开始处理，共找到 {len(image_files)} 张图片...\n" + "-"*50)

    batch_arts = {}

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(input_folder, filename)

        try:
            # 1. 转换为 ASCII 字符
            ascii_text, orig_w, orig_h, out_h = convert_image_to_ascii_str(
                image_path,
                output_width=output_width,
                font_aspect_ratio=font_aspect_ratio,
                dark_mode=dark_mode
            )

            batch_arts[filename] = ascii_text

            print(f"[{idx}/{len(image_files)}] ✅ {filename} (原图 {orig_w}x{orig_h} -> 网格 {output_width}x{out_h})")

        except Exception as e:
            print(f"[{idx}/{len(image_files)}] ❌ 处理 {filename} 失败: {e}")

    if batch_arts:
        all_gallery_path = os.path.join(output_folder, "all_ascii_art.html")
        save_batch_html(batch_arts, all_gallery_path, dark_mode=dark_mode)
        print("-" * 50)
        print(f"🎉 全部处理完成！总网页:")
        print(f"🌐 {os.path.abspath(all_gallery_path)}")


# ==================== 运行入口 ====================
if __name__ == "__main__":
    batch_process_folder(
        input_folder=DEFAULT_INPUT,   # 项目根目录的 input 文件夹
        output_folder=DEFAULT_OUTPUT, # 项目根目录的 output 文件夹
        output_width=150,             # 每一张字符画的横向分辨率
        font_aspect_ratio=0.75,       # 字体高宽比参数
        dark_mode=True                # 黑底白字模式
    )
"""Giao diện Gradio. Hugging Face Spaces tự chạy file này."""
import logging

import gradio as gr

from legal_qa.rag import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
pipeline = RAGPipeline()


def respond(message, history):
    # Mỗi câu hỏi tra cứu độc lập (không dùng lịch sử chat)
    yield from pipeline.stream_answer(message)


demo = gr.ChatInterface(
    fn=respond,
    title="Trợ lý tra cứu Bộ luật Lao động 2019",
    description=("Hỏi đáp có trích dẫn điều luật (45/2019/QH14). Mỗi câu hỏi được tra cứu độc lập. "
                 "Thông tin chỉ mang tính tham khảo, không thay thế tư vấn pháp lý."),
    examples=["Vợ sinh con được nghỉ mấy tháng?",
              "Thử việc tối đa bao lâu?",
              "Muốn nghỉ việc phải báo trước mấy ngày?",
              "Công ty có được giữ bằng gốc của nhân viên không?"],
    cache_examples=False,  # không tự chạy ví dụ lúc build để khỏi tốn quota
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()

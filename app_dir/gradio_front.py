import gradio as gr

from app_dir.back.app_logic import get_pipeline_prediction, get_matrix_and_hist_plot

# demo = gr.Interface(
#     fn=get_pipeline_prediction,
#     inputs=[gr.Text(label="channel id", type="text"), gr.Slider(0, 1, step=0.05, value=0.8), gr.Number()],
#     outputs=[gr.DataFrame(), gr.File(file_types=[".html"])]
# )


with gr.Blocks() as demo:
    # gr.Markdown("Flip text or image files using this demo.")

    # in
    channel_id = gr.Text(label="channel id", type="text", value='UCansLl8T6imFHqOhfXoAHmg', info="WRITE OWN CHANNEL ID")
    amount_of_videos = gr.Slider(0, 2000, step=1, value=200, label="amount_of_videos",
                                 info="WRITE AMOUNT OF VIDEOS OR BIGGER")

    rate = gr.Slider(0, 1, step=0.05, value=0.8,
                     info="CHOOSE RATE WITH 'Analyze embedding matrix' SECTION. "
                          "'rate' - INFLUENCE ON WHICH NODES HAs EDGES")

    with gr.Tab("Main"):
        # in
        # rate = gr.Slider(0, 1, step=0.05, value=0.8)
        # out
        # with gr.Row():
        table_output = gr.DataFrame()
        graph_file_output = gr.File(file_types=[".html"])
        # butt
        group_videos_from_channel = gr.Button("group_videos_from_channel")

    with gr.Tab("Analyze embedding matrix"):
        # out
        with gr.Row():
            matrx = gr.Text()
            hist_plot = gr.Plot()
        # butt
        show_matrix_and_hist_plot = gr.Button("show_matrix_and_hist_plot")

    # with gr.Accordion("Open for More!", open=False):
    #     gr.Markdown("Look at me...")
    #     temp_slider = gr.Slider(
    #         minimum=0.0,
    #         maximum=1.0,
    #         value=0.1,
    #         step=0.1,
    #         interactive=True,
    #         label="Slide me",
    #     )
    #     temp_slider.change(lambda x: x, [temp_slider])

    group_videos_from_channel.click(get_pipeline_prediction,
                                    inputs=[channel_id, rate, amount_of_videos],
                                    outputs=[table_output, graph_file_output])
    show_matrix_and_hist_plot.click(get_matrix_and_hist_plot,
                                    inputs=[channel_id, amount_of_videos],
                                    outputs=[matrx, hist_plot])

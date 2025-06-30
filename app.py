# --- Chat Interface ---
st.subheader("💬 Chat Interface")
display_mode = st.radio("Select display mode:", ["Both", "Chart only", "Text only"], horizontal=True)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question like 'Top 5 drugs by spending' or follow up with context")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Analyzing..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.chat_history,
            functions=functions,
            function_call="auto"
        )
        msg = response.choices[0].message
        st.session_state.chat_history.append(msg)
        with st.chat_message("assistant"):
            if hasattr(msg, "function_call") and msg.function_call:
                fname = msg.function_call.name
                args = json.loads(msg.function_call.arguments)
                try:
                    result = globals()[fname](**args)
                    if isinstance(result, dict):
                        if display_mode in ["Both", "Chart only"]:
                            series = pd.Series(result)
                            if len(series) > 1:
                                fig, ax = plt.subplots(figsize=(8, 4))
                                series.plot(kind='bar', ax=ax)
                                ax.set_title(f"{fname} on {args.get('column', '')}")
                                plt.xticks(rotation=30, ha='right')
                                st.pyplot(fig)

                        if display_mode in ["Both", "Text only"]:
                            output_lines = []
                            for k, v in result.items():
                                if isinstance(v, (int, float)) and v > 1000:
                                    output_lines.append(f"{k.strip()}: ${v:,.2f}")
                                else:
                                    output_lines.append(f"{k.strip()}: {v}")
                            st.text("\n".join(output_lines))
                    else:
                        st.write(result)
                    st.session_state.conversation_log.append({
                        "question": user_input,
                        "function": fname,
                        "args": args,
                        "result": result
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.markdown(msg.content)
                st.session_state.conversation_log.append({
                    "question": user_input,
                    "answer": msg.content
                })

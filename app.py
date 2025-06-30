# Add timeout + safe fallback for chat response
with st.spinner("Analyzing..."):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.chat_history,
            functions=functions,
            function_call="auto",
            timeout=30  # Add timeout to prevent hanging forever
        )
        msg = response.choices[0].message
        st.session_state.chat_history.append(msg)

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
                st.error(f"Function error: {e}")
        else:
            if msg.content:
                st.chat_message("assistant").markdown(msg.content)
                st.session_state.conversation_log.append({
                    "question": user_input,
                    "answer": msg.content
                })
            else:
                st.warning("🤖 Assistant did not return a response. Please try rephrasing your question.")
    except Exception as e:
        st.error(f"Chat request failed: {e}")

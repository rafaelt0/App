"""Session-scoped anonymous identity for portfolio and watchlist data."""
import uuid

import streamlit as st


def get_browser_uid() -> str:
    """Return a random identity for this Streamlit session, never from the URL."""
    if "uid" in st.query_params:
        del st.query_params["uid"]

    # Do not reuse _browser_uid: older versions populated it from the URL token.
    st.session_state.pop("_browser_uid", None)
    if "_session_uid" not in st.session_state:
        st.session_state["_session_uid"] = uuid.uuid4().hex
    return st.session_state["_session_uid"]

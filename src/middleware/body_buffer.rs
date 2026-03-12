use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use http_body_util::BodyExt;

/// A clone-able snapshot of the request body, stored in request extensions.
///
/// Inserted by [`buffer_body_middleware`] so that every downstream
/// middleware and the final handler can access the original body
/// without consuming it from the request stream.
#[derive(Clone, Debug)]
pub struct BufferedBody(pub Bytes);

/// Body-buffering middleware (outermost layer).
///
/// Reads the request body **once**, stores a copy in
/// [`BufferedBody`] via request extensions, then re-attaches the
/// bytes as the request body so that standard Axum extractors
/// continue to work.
///
/// All downstream layers should prefer reading from
/// `req.extensions().get::<BufferedBody>()` instead of consuming
/// the body stream directly.
pub async fn buffer_body_middleware(request: Request, next: Next) -> Response {
    let (parts, body) = request.into_parts();

    let bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "Failed to read request body").into_response();
        }
    };

    let buffered = BufferedBody(bytes.clone());

    // Re-assemble the request with the body bytes restored **and**
    // the buffered copy in extensions.
    let mut request = Request::from_parts(parts, Body::from(bytes));
    request.extensions_mut().insert(buffered);

    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{middleware as axum_mw, routing::post, Router};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// Handler that echoes the buffered body from extensions.
    async fn echo_buffered(req: Request) -> Response {
        match req.extensions().get::<BufferedBody>() {
            Some(buf) => {
                let text = String::from_utf8_lossy(&buf.0).to_string();
                (StatusCode::OK, text).into_response()
            }
            None => (StatusCode::INTERNAL_SERVER_ERROR, "no buffered body").into_response(),
        }
    }

    /// Handler that reads the body stream (should still work).
    async fn echo_stream(req: Request) -> Response {
        match req.into_body().collect().await {
            Ok(collected) => {
                let text = String::from_utf8_lossy(&collected.to_bytes()).to_string();
                (StatusCode::OK, text).into_response()
            }
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "stream read failed").into_response(),
        }
    }

    #[tokio::test]
    async fn buffered_body_available_in_extensions() {
        let app = Router::new()
            .route("/test", post(echo_buffered))
            .layer(axum_mw::from_fn(buffer_body_middleware));

        let req = Request::builder()
            .method("POST")
            .uri("/test")
            .body(Body::from(r#"{"prompt":"hello"}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], br#"{"prompt":"hello"}"#);
    }

    #[tokio::test]
    async fn body_stream_still_readable() {
        let app = Router::new()
            .route("/test", post(echo_stream))
            .layer(axum_mw::from_fn(buffer_body_middleware));

        let req = Request::builder()
            .method("POST")
            .uri("/test")
            .body(Body::from("raw prompt"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"raw prompt");
    }

    #[tokio::test]
    async fn empty_body_produces_empty_buffer() {
        let app = Router::new()
            .route("/test", post(echo_buffered))
            .layer(axum_mw::from_fn(buffer_body_middleware));

        let req = Request::builder()
            .method("POST")
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert!(body.is_empty());
    }
}

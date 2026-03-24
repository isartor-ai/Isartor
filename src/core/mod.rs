pub mod cache_scope;
pub mod ports;
pub mod prompt;
pub mod retry;

// ── Network utility helpers ───────────────────────────────────────────

/// Returns `true` when `url` resolves to an internal / loopback destination
/// and `false` when it appears to be an external internet endpoint.
///
/// Used to decide whether an OTel or L3 endpoint would "phone home" and
/// should therefore be suppressed in offline mode.
///
/// The check uses proper hostname extraction via URL parsing so that
/// hostnames like `my-localhost.corp` are NOT confused with `localhost`.
pub fn is_internal_endpoint(url: &str) -> bool {
    // Parse the URL to extract just the hostname component.
    // We accept both `scheme://host:port/path` and bare `host:port` forms.
    let host = extract_host(url);

    // Loopback / link-local literals.
    if host == "localhost" || host == "127.0.0.1" || host == "::1" || host.starts_with("127.") {
        return true;
    }

    // Private IPv4 ranges (RFC 1918): 10.x, 172.16-31.x, 192.168.x.
    if is_private_ipv4(&host) {
        return true;
    }

    // Kubernetes / internal DNS suffixes.
    if host.ends_with(".svc")
        || host.ends_with(".svc.cluster.local")
        || host.ends_with(".local")
        || host.ends_with(".internal")
        || host.ends_with(".corp")
        || host.ends_with(".lan")
    {
        return true;
    }

    false
}

/// Extract the hostname from a URL string.
/// Handles `http://host:port/path`, `grpc://host:port`, and bare `host:port`.
fn extract_host(url: &str) -> String {
    // Strip scheme (e.g. "http://", "https://", "grpc://").
    let without_scheme = if let Some(idx) = url.find("://") {
        &url[idx + 3..]
    } else {
        url
    };
    // Strip optional `userinfo@` (e.g. "user:pass@" or ":password@") from the authority.
    let without_userinfo = if let Some(at) = without_scheme.rfind('@') {
        &without_scheme[at + 1..]
    } else {
        without_scheme
    };
    // Strip path.
    let without_path = without_userinfo
        .split('/')
        .next()
        .unwrap_or(without_userinfo);
    // Strip port.
    // Handle IPv6 addresses like `[::1]:4317` specially.
    if without_path.starts_with('[') {
        // IPv6 literal — everything up to the closing `]`.
        if let Some(end) = without_path.find(']') {
            return without_path[1..end].to_lowercase();
        }
    }
    // For IPv4 / hostname, strip the optional `:port`.
    without_path
        .split(':')
        .next()
        .unwrap_or(without_path)
        .to_lowercase()
}

/// Returns `true` if the string is a private-range IPv4 address.
fn is_private_ipv4(host: &str) -> bool {
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() != 4 {
        return false;
    }
    let octets: Vec<u8> = parts.iter().filter_map(|p| p.parse::<u8>().ok()).collect();
    if octets.len() != 4 {
        return false;
    }
    // 10.0.0.0/8
    if octets[0] == 10 {
        return true;
    }
    // 172.16.0.0/12
    if octets[0] == 172 && (16..=31).contains(&octets[1]) {
        return true;
    }
    // 192.168.0.0/16
    if octets[0] == 192 && octets[1] == 168 {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn localhost_is_internal() {
        assert!(is_internal_endpoint("http://localhost:4317"));
        assert!(is_internal_endpoint("localhost:4317"));
    }

    #[test]
    fn loopback_ipv4_is_internal() {
        assert!(is_internal_endpoint("http://127.0.0.1:4317"));
        assert!(is_internal_endpoint("127.0.0.1:4317"));
        assert!(is_internal_endpoint("127.1.2.3:9000"));
    }

    #[test]
    fn svc_suffix_is_internal() {
        assert!(is_internal_endpoint("http://otel-collector.svc:4317"));
        assert!(is_internal_endpoint("redis.svc.cluster.local:6379"));
    }

    #[test]
    fn external_hostname_is_not_internal() {
        assert!(!is_internal_endpoint("https://api.openai.com"));
        assert!(!is_internal_endpoint("https://api.anthropic.com"));
        assert!(!is_internal_endpoint(
            "http://my-localhost.example.com:4317"
        ));
    }

    #[test]
    fn private_ipv4_is_internal() {
        assert!(is_internal_endpoint("http://10.0.0.1:4317"));
        assert!(is_internal_endpoint("http://172.16.0.1:4317"));
        assert!(is_internal_endpoint("http://192.168.1.1:4317"));
    }

    #[test]
    fn public_ipv4_is_not_internal() {
        assert!(!is_internal_endpoint("http://8.8.8.8:53"));
    }
}

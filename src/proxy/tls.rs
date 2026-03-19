//! Local CA generation and leaf certificate signing for TLS MITM.
//!
//! On first use, generates a self-signed root CA stored at
//! `~/.isartor/ca/isartor-ca.pem` and `~/.isartor/ca/isartor-ca-key.pem`.
//! Subsequent runs load the existing CA from disk.
//!
//! Provides `server_config_for_host` to generate ephemeral leaf certificates
//! signed by the CA for a given hostname, used by the CONNECT proxy to
//! terminate TLS connections from clients.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use rcgen::{
    BasicConstraints, CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, IsCa,
    KeyPair, KeyUsagePurpose, SanType,
};
use rustls::ServerConfig;
use tokio_rustls::rustls;

const CA_DIR: &str = ".isartor/ca";
const CA_CERT_FILE: &str = "isartor-ca.pem";
const CA_KEY_FILE: &str = "isartor-ca-key.pem";

/// Holds the loaded or generated CA certificate and key pair.
pub struct IsartorCa {
    #[allow(dead_code)]
    ca_cert_pem: String,
    #[allow(dead_code)]
    ca_key_pem: String,
    ca_cert_path: PathBuf,
    // Parsed for signing leaf certs
    ca_key_pair: KeyPair,
    ca_cert_params: rcgen::Certificate,
}

impl IsartorCa {
    /// Load the CA from `~/.isartor/ca/` or generate a new one if missing.
    pub fn load_or_generate() -> Result<Self> {
        let home = dirs::home_dir().context("Could not determine home directory")?;
        let ca_dir = home.join(CA_DIR);
        let cert_path = ca_dir.join(CA_CERT_FILE);
        let key_path = ca_dir.join(CA_KEY_FILE);

        if cert_path.exists() && key_path.exists() {
            tracing::info!(path = %cert_path.display(), "Loading existing Isartor CA");
            let ca_cert_pem =
                std::fs::read_to_string(&cert_path).context("Failed to read CA cert")?;
            let ca_key_pem = std::fs::read_to_string(&key_path).context("Failed to read CA key")?;

            let ca_key_pair =
                KeyPair::from_pem(&ca_key_pem).context("Failed to parse CA key pair")?;
            let ca_cert_params = CertificateParams::from_ca_cert_pem(&ca_cert_pem)
                .context("Failed to parse CA cert params")?;
            let ca_cert = ca_cert_params
                .self_signed(&ca_key_pair)
                .context("Failed to re-sign CA cert for internal use")?;

            Ok(Self {
                ca_cert_pem,
                ca_key_pem,
                ca_cert_path: cert_path,
                ca_key_pair,
                ca_cert_params: ca_cert,
            })
        } else {
            tracing::info!("Generating new Isartor CA certificate");
            std::fs::create_dir_all(&ca_dir)
                .with_context(|| format!("Failed to create {}", ca_dir.display()))?;

            let ca_key_pair = KeyPair::generate().context("Failed to generate CA key pair")?;

            let mut ca_params = CertificateParams::default();
            ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
            ca_params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];

            let mut dn = DistinguishedName::new();
            dn.push(DnType::CommonName, "Isartor Local CA");
            dn.push(DnType::OrganizationName, "Isartor");
            ca_params.distinguished_name = dn;

            // 10 year validity
            let not_before = time::OffsetDateTime::now_utc();
            let not_after = not_before + time::Duration::days(3650);
            ca_params.not_before = not_before;
            ca_params.not_after = not_after;

            let ca_cert = ca_params
                .self_signed(&ca_key_pair)
                .context("Failed to self-sign CA certificate")?;

            let ca_cert_pem = ca_cert.pem();
            let ca_key_pem = ca_key_pair.serialize_pem();

            std::fs::write(&cert_path, &ca_cert_pem)
                .with_context(|| format!("Failed to write {}", cert_path.display()))?;
            std::fs::write(&key_path, &ca_key_pem)
                .with_context(|| format!("Failed to write {}", key_path.display()))?;

            // Restrict key file permissions on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600)).ok();
            }

            tracing::info!(
                cert = %cert_path.display(),
                key = %key_path.display(),
                "Isartor CA generated"
            );

            Ok(Self {
                ca_cert_pem,
                ca_key_pem,
                ca_cert_path: cert_path,
                ca_key_pair,
                ca_cert_params: ca_cert,
            })
        }
    }

    /// Path to the CA PEM certificate file.
    pub fn ca_cert_path(&self) -> &Path {
        &self.ca_cert_path
    }

    /// Return the path to a combined PEM bundle (system CAs + Isartor CA).
    ///
    /// Required so that non-Node.js clients (Go, Python, curl) trust both
    /// the Isartor MITM certificates **and** real upstream certificates for
    /// tunnelled connections.  The bundle is written once to
    /// `~/.isartor/ca/combined-ca.pem` and refreshed on every call.
    pub fn combined_ca_bundle_path(&self) -> Result<PathBuf> {
        let bundle_path = self
            .ca_cert_path
            .parent()
            .context("CA cert has no parent directory")?
            .join("combined-ca.pem");

        let system_pem = read_system_ca_bundle()?;

        let combined = format!("{}\n{}", system_pem.trim_end(), self.ca_cert_pem.trim_end());
        std::fs::write(&bundle_path, &combined)
            .with_context(|| format!("Failed to write {}", bundle_path.display()))?;

        tracing::debug!(path = %bundle_path.display(), "Combined CA bundle written");
        Ok(bundle_path)
    }

    /// Generate a `rustls::ServerConfig` with a leaf certificate for the given hostname,
    /// signed by this CA.
    pub fn server_config_for_host(&self, hostname: &str) -> Result<Arc<ServerConfig>> {
        let leaf_key = KeyPair::generate().context("Failed to generate leaf key pair")?;

        let mut leaf_params = CertificateParams::default();
        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, hostname);
        leaf_params.distinguished_name = dn;
        leaf_params.subject_alt_names = vec![SanType::DnsName(hostname.try_into()?)];
        leaf_params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];

        // 1 day validity
        let not_before = time::OffsetDateTime::now_utc();
        let not_after = not_before + time::Duration::days(1);
        leaf_params.not_before = not_before;
        leaf_params.not_after = not_after;

        let leaf_cert = leaf_params
            .signed_by(&leaf_key, &self.ca_cert_params, &self.ca_key_pair)
            .context("Failed to sign leaf certificate")?;

        let leaf_cert_der = rustls::pki_types::CertificateDer::from(leaf_cert.der().to_vec());
        let ca_cert_der =
            rustls::pki_types::CertificateDer::from(self.ca_cert_params.der().to_vec());
        let leaf_key_der =
            rustls::pki_types::PrivateKeyDer::try_from(leaf_key.serialized_der().to_vec())
                .map_err(|e| anyhow::anyhow!("Failed to convert leaf key to DER: {e}"))?;

        let mut config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![leaf_cert_der, ca_cert_der], leaf_key_der)
            .context("Failed to build rustls ServerConfig")?;

        // Only offer HTTP/1.1 — the proxy's request parser is text-based
        // and cannot handle HTTP/2 binary frames.
        config.alpn_protocols = vec![b"http/1.1".to_vec()];

        Ok(Arc::new(config))
    }
}

/// Read the platform's default CA bundle as a PEM string.
fn read_system_ca_bundle() -> Result<String> {
    // Prefer an explicit override first (rarely set, but useful in containers).
    if let Ok(path) = std::env::var("SSL_CERT_FILE") {
        let p = std::path::Path::new(&path);
        if p.exists() {
            return std::fs::read_to_string(p)
                .with_context(|| format!("Failed to read SSL_CERT_FILE={path}"));
        }
    }

    // Well-known system paths (macOS, Debian/Ubuntu, RHEL/Fedora, Alpine).
    let candidates = [
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/ca-bundle.pem",
    ];

    for path in candidates {
        let p = std::path::Path::new(path);
        if p.exists() {
            return std::fs::read_to_string(p)
                .with_context(|| format!("Failed to read system CA bundle at {path}"));
        }
    }

    anyhow::bail!(
        "Could not locate the system CA bundle. \
         Set SSL_CERT_FILE to the path of your CA bundle."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn install_crypto_provider() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }

    #[test]
    fn test_load_or_generate_ca() {
        install_crypto_provider();
        let ca = IsartorCa::load_or_generate().expect("CA generation should succeed");
        assert!(ca.ca_cert_path().exists());
        assert!(!ca.ca_cert_pem.is_empty());
    }

    #[test]
    fn test_server_config_for_host() {
        install_crypto_provider();
        let ca = IsartorCa::load_or_generate().expect("CA should load");
        let config = ca
            .server_config_for_host("example.com")
            .expect("Leaf cert generation should succeed");
        assert!(!config.alpn_protocols.is_empty());
    }
}

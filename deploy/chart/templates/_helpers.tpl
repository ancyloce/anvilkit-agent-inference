{{- define "anvilkit-agent-inference.name" -}}
anvilkit-agent-inference
{{- end }}

{{- define "anvilkit-agent-inference.fullname" -}}
{{- if contains "anvilkit-agent-inference" .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-anvilkit-agent-inference" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end }}

{{- define "anvilkit-agent-inference.labels" -}}
app.kubernetes.io/name: {{ include "anvilkit-agent-inference.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "anvilkit-agent-inference.selectorLabels" -}}
app.kubernetes.io/name: {{ include "anvilkit-agent-inference.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "anvilkit-agent-inference.serviceAccountName" -}}
{{- default (include "anvilkit-agent-inference.fullname" .) .Values.serviceAccount.name -}}
{{- end }}

{{- define "anvilkit-agent-inference.image" -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else if .Values.development.enabled -}}
{{- printf "%s:%s" .Values.image.repository (required "image.tag or image.digest is required" .Values.image.tag) -}}
{{- else -}}
{{- fail "image.digest is required outside development: a tag is mutable and only a digest pins the image (RB-S10); the tag renders only with development.enabled: true" -}}
{{- end -}}
{{- end }}

{{- define "anvilkit-agent-inference.tlsSecret" -}}
{{- default (printf "%s-tls" (include "anvilkit-agent-inference.fullname" .)) .Values.tls.tlsSecret -}}
{{- end }}

{{- define "anvilkit-agent-inference.require" -}}
{{- if not (has .Values.tls.mode (list "tls" "mtls" "development")) }}
{{- fail "tls.mode must be tls, mtls or development" }}
{{- end }}
{{- if and (eq .Values.tls.mode "development") (not .Values.development.enabled) }}
{{- fail "tls.mode development is DEVELOPMENT_ONLY: plaintext HTTP renders only with development.enabled: true" }}
{{- end }}
{{- if and (eq .Values.tls.mode "tls") (not .Values.development.enabled) }}
{{- fail "tls.mode tls authenticates no caller and is DEVELOPMENT_ONLY: it renders only with development.enabled: true; outside development use tls.mode mtls with tls.allowedPeers and listen.probePort (RB-A17)" }}
{{- end }}
{{- if and (has .Values.tls.mode (list "tls" "mtls")) .Values.tls.certificate.create (not .Values.tls.certificate.issuerRef.name) }}
{{- fail "tls.certificate.issuerRef.name is required to issue the listener certificate (or set certificate.create false and tls.tlsSecret)" }}
{{- end }}
{{- if and (has .Values.tls.mode (list "tls" "mtls")) (not .Values.tls.certificate.create) (not .Values.tls.tlsSecret) }}
{{- fail "tls.tlsSecret is required when tls.certificate.create is false" }}
{{- end }}
{{- if and (eq .Values.tls.mode "mtls") (not .Values.tls.allowedPeers) }}
{{- fail "tls.allowedPeers is required under tls.mode mtls: the SPIFFE IDs that may call (Knowledge's workload identity)" }}
{{- end }}
{{- if and (eq .Values.tls.mode "mtls") (not (int .Values.listen.probePort)) }}
{{- fail "listen.probePort is required under tls.mode mtls: kubelet probes present no client certificate" }}
{{- end }}
{{- if and (ne .Values.tls.mode "mtls") .Values.tls.allowedPeers }}
{{- fail "tls.allowedPeers authenticates callers only under tls.mode mtls" }}
{{- end }}
{{- if and (int .Values.listen.probePort) (eq (int .Values.listen.probePort) (int .Values.listen.port)) }}
{{- fail "listen.probePort must differ from listen.port" }}
{{- end }}
{{- end }}

{{/* The probes reach the plaintext probe listener when listen.probePort is
set (required under mtls), otherwise the main listener. */}}
{{- define "anvilkit-agent-inference.probeScheme" -}}
{{- if int .Values.listen.probePort }}HTTP{{ else if eq .Values.tls.mode "tls" }}HTTPS{{ else }}HTTP{{ end -}}
{{- end }}

{{- define "anvilkit-agent-inference.probePort" -}}
{{- if int .Values.listen.probePort }}probe{{ else }}http{{ end -}}
{{- end }}

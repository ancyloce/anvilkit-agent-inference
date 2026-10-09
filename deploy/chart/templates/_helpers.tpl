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
{{- else -}}
{{- printf "%s:%s" .Values.image.repository (required "image.tag or image.digest is required" .Values.image.tag) -}}
{{- end -}}
{{- end }}

{{- define "anvilkit-agent-inference.tlsSecret" -}}
{{- default (printf "%s-tls" (include "anvilkit-agent-inference.fullname" .)) .Values.tls.tlsSecret -}}
{{- end }}

{{- define "anvilkit-agent-inference.require" -}}
{{- if not (has .Values.tls.mode (list "tls" "development")) }}
{{- fail "tls.mode must be tls or development" }}
{{- end }}
{{- if and (eq .Values.tls.mode "development") (not .Values.development.enabled) }}
{{- fail "tls.mode development is DEVELOPMENT_ONLY: plaintext HTTP renders only with development.enabled: true" }}
{{- end }}
{{- if and (eq .Values.tls.mode "tls") .Values.tls.certificate.create (not .Values.tls.certificate.issuerRef.name) }}
{{- fail "tls.certificate.issuerRef.name is required to issue the listener certificate (or set certificate.create false and tls.tlsSecret)" }}
{{- end }}
{{- if and (eq .Values.tls.mode "tls") (not .Values.tls.certificate.create) (not .Values.tls.tlsSecret) }}
{{- fail "tls.tlsSecret is required when tls.certificate.create is false" }}
{{- end }}
{{- end }}

{{- define "anvilkit-agent-inference.probeScheme" -}}
{{- if eq .Values.tls.mode "tls" }}HTTPS{{ else }}HTTP{{ end -}}
{{- end }}

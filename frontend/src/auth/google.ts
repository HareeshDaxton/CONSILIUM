// Google Identity Services (GIS) loader. The browser receives a Google ID
// token (`credential`) and hands it to our API (POST /v1/auth/google), which
// verifies it and returns a CONSILIUM session token. The UI never treats the
// Google token as authorization by itself.

const GIS_SRC = "https://accounts.google.com/gsi/client";

export interface CredentialResponse {
  credential: string;
  select_by?: string;
}

export interface GoogleButtonOptions {
  type?: "standard" | "icon";
  theme?: "outline" | "filled_blue" | "filled_black";
  size?: "large" | "medium" | "small";
  text?: "signin_with" | "signup_with" | "continue_with" | "signin";
  shape?: "rectangular" | "pill" | "circle" | "square";
  logo_alignment?: "left" | "center";
  width?: number;
}

export interface GoogleAccountsId {
  initialize(config: {
    client_id: string;
    callback: (response: CredentialResponse) => void;
    auto_select?: boolean;
    cancel_on_tap_outside?: boolean;
    use_fedcm_for_prompt?: boolean;
    context?: "signin" | "signup" | "use";
  }): void;
  renderButton(parent: HTMLElement, options: GoogleButtonOptions): void;
  prompt(): void;
  cancel(): void;
  disableAutoSelect(): void;
}

declare global {
  interface Window {
    google?: { accounts: { id: GoogleAccountsId } };
  }
}

let loading: Promise<GoogleAccountsId> | null = null;

/** Injects the GIS script once and resolves with `google.accounts.id`. */
export function loadGoogleIdentity(): Promise<GoogleAccountsId> {
  if (window.google?.accounts?.id) return Promise.resolve(window.google.accounts.id);
  loading ??= new Promise<GoogleAccountsId>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = GIS_SRC;
    script.async = true;
    script.defer = true;
    script.onload = () =>
      window.google?.accounts?.id
        ? resolve(window.google.accounts.id)
        : reject(new Error("GIS_UNAVAILABLE"));
    script.onerror = () => {
      loading = null;
      reject(new Error("GIS_LOAD_FAILED"));
    };
    document.head.appendChild(script);
  });
  return loading;
}

/** After an explicit sign-out, stop One Tap from signing the user straight back in. */
export function disableGoogleAutoSelect(): void {
  window.google?.accounts?.id?.disableAutoSelect();
}

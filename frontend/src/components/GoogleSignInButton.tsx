import { useEffect, useRef, useState } from "react";

import { loadGoogleIdentity } from "../auth/google";

// Renders Google's own sign-in button (GIS) and, for returning users, the One
// Tap prompt with auto-select — so a page reload (which clears the in-memory
// session) signs a clinician back in without a click.
export function GoogleSignInButton({
  clientId,
  onCredential,
  autoPrompt = true,
}: {
  clientId: string;
  onCredential: (credential: string) => void;
  autoPrompt?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const callback = useRef(onCredential);
  callback.current = onCredential;
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    loadGoogleIdentity()
      .then((gis) => {
        if (cancelled || !host.current) return;
        gis.initialize({
          client_id: clientId,
          callback: (r) => callback.current(r.credential),
          auto_select: true,
          cancel_on_tap_outside: true,
          use_fedcm_for_prompt: true,
          context: "signin",
        });
        gis.renderButton(host.current, {
          type: "standard",
          theme: "filled_black",
          size: "large",
          text: "continue_with",
          shape: "pill",
          logo_alignment: "left",
          width: 320,
        });
        if (autoPrompt) gis.prompt();
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
      window.google?.accounts?.id?.cancel();
    };
  }, [clientId, autoPrompt]);

  if (failed) {
    return (
      <p className="field-error" role="alert">
        Google sign-in could not load. Check your connection and try again.
      </p>
    );
  }
  return <div ref={host} className="google-button" data-testid="google-button" />;
}

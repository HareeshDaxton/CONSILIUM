import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// jsdom has no object URLs; the overlay viewer turns server PNG blobs into them.
let objectUrls = 0;
if (!("createObjectURL" in URL)) {
  Object.defineProperty(URL, "createObjectURL", { value: () => `blob:test/${++objectUrls}`, writable: true });
  Object.defineProperty(URL, "revokeObjectURL", { value: () => undefined, writable: true });
}

afterEach(() => {
  cleanup();
});

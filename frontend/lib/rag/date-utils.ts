const MONTHS: Record<string, number> = {
  january: 1,
  jan: 1,
  february: 2,
  feb: 2,
  march: 3,
  mar: 3,
  april: 4,
  apr: 4,
  may: 5,
  june: 6,
  jun: 6,
  july: 7,
  jul: 7,
  august: 8,
  aug: 8,
  september: 9,
  sep: 9,
  sept: 9,
  october: 10,
  oct: 10,
  november: 11,
  nov: 11,
  december: 12,
  dec: 12,
};

export type DateMetadata = {
  raw: string;
  start?: number;
  end?: number;
  isPresent: boolean;
};

export function extractDateMetadata(text: string): DateMetadata | undefined {
  const rangeMatch = text.match(
    /\b([A-Za-z]{3,9}\s+\d{4}|\d{4})\s*[–-]\s*(Present|Current|[A-Za-z]{3,9}\s+\d{4}|\d{4})\b/i,
  );

  if (rangeMatch) {
    const start = parseDateValue(rangeMatch[1]);
    const isPresent = /present|current/i.test(rangeMatch[2]);
    const end = isPresent ? Number.MAX_SAFE_INTEGER : parseDateValue(rangeMatch[2]);

    return {
      raw: rangeMatch[0],
      start,
      end,
      isPresent,
    };
  }

  const singleDate = text.match(/\b([A-Za-z]{3,9}\s+\d{4}|\d{4})\b/);

  if (singleDate) {
    const value = parseDateValue(singleDate[1]);

    return {
      raw: singleDate[0],
      start: value,
      end: value,
      isPresent: false,
    };
  }

  if (/\b(Present|Current)\b/i.test(text)) {
    return {
      raw: "Present",
      end: Number.MAX_SAFE_INTEGER,
      isPresent: true,
    };
  }

  return undefined;
}

export function compareRecent(first?: DateMetadata, second?: DateMetadata): number {
  const firstEnd = first?.end ?? first?.start ?? 0;
  const secondEnd = second?.end ?? second?.start ?? 0;

  if (secondEnd !== firstEnd) {
    return secondEnd - firstEnd;
  }

  return (second?.start ?? 0) - (first?.start ?? 0);
}

function parseDateValue(value: string): number | undefined {
  const yearOnly = value.match(/^\d{4}$/);

  if (yearOnly) {
    return Number(value) * 100 + 1;
  }

  const match = value.toLowerCase().match(/^([a-z]{3,9})\s+(\d{4})$/);

  if (!match) {
    return undefined;
  }

  const month = MONTHS[match[1]];

  if (!month) {
    return undefined;
  }

  return Number(match[2]) * 100 + month;
}
